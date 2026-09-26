import numpy as np

class ConstraintManager:
    """
    Manages and applies constraints to the parameter vector (theta) during optimization.
    
    This ensures that tuned parameters remain within logical bounds (e.g., bonuses >= 0)
    and maintain logical relationships (e.g., higher rank passed pawn = higher bonus).
    """
    def __init__(self, param_manager):
        self.pm = param_manager
        self.constraints = []

    def add_range(self, param_name, min_val=None, max_val=None):
        """
        Constrains a parameter (or array of parameters) to be within [min_val, max_val].
        
        Args:
            param_name (str): The name of the parameter in ParameterManager.
            min_val (float, optional): The lower bound.
            max_val (float, optional): The upper bound.
        """
        self.constraints.append({
            'type': 'range',
            'name': param_name,
            'min': min_val,
            'max': max_val
        })

    def add_range_on_absolute_index(self, abs_idx, min_val=None, max_val=None):
        """
        Constrains a specific absolute theta index to be within [min_val, max_val].
        Used to pin individual elements like King material to 0.
        
        Args:
            abs_idx (int): Absolute index in the theta vector.
            min_val (float, optional): The lower bound.
            max_val (float, optional): The upper bound.
        """
        self.constraints.append({
            'type': 'range_absolute',
            'abs_idx': abs_idx,
            'min': min_val,
            'max': max_val
        })

    def add_monotonic(self, param_name, axis=0, direction='increasing', end_idx=None):
        """
        Constrains a parameter array to be monotonic along a specific axis.
        
        Args:
            param_name (str): The name of the parameter.
            axis (int): The axis to enforce monotonicity on (0 for rows/first dim).
            direction (str): 'increasing' (v[i] <= v[i+1]) or 'decreasing' (v[i] >= v[i+1]).
            end_idx (int, optional): Only enforce monotonicity for elements [:end_idx] along axis.
                                     Useful to exclude trailing elements (e.g., King in material array).
        """
        self.constraints.append({
            'type': 'monotonic',
            'name': param_name,
            'axis': axis,
            'direction': direction,
            'end_idx': end_idx
        })

    def add_fixed_mean(
        self, param_name, relative_indices, target_mean, quantized=False
    ):
        """Keep the selected entries' mean fixed while allowing shape changes.

        This removes additive gauge freedom (for example material versus a
        constant shift of one PST row) without changing the starting table.
        Indices are relative to the flattened parameter.
        """
        self.constraints.append({
            'type': 'fixed_mean',
            'name': param_name,
            'relative_indices': tuple(int(i) for i in relative_indices),
            'target_mean': float(target_mean),
            'quantized': bool(quantized),
        })

    def apply(self, theta, active_mask=None):
        """
        Applies all registered constraints to the theta vector in-place.

        ``active_mask`` is optional. Fixed-mean projections use it to avoid
        moving parameters frozen by the SPSA tune/exclude mask.
        """
        for c in self.constraints:
            self._apply_constraint(theta, c, active_mask=active_mask)

    def _absolute_bounds(self, param_name, absolute_indices):
        """Return the intersection of registered range bounds for indices."""
        lower = np.full(len(absolute_indices), -np.inf, dtype=np.float64)
        upper = np.full(len(absolute_indices), np.inf, dtype=np.float64)
        positions = {int(idx): pos for pos, idx in enumerate(absolute_indices)}

        for c in self.constraints:
            ctype = c['type']
            if ctype == 'range' and c['name'] == param_name:
                if c['min'] is not None:
                    lower[:] = np.maximum(lower, float(c['min']))
                if c['max'] is not None:
                    upper[:] = np.minimum(upper, float(c['max']))
            elif ctype == 'range_absolute':
                pos = positions.get(int(c['abs_idx']))
                if pos is None:
                    continue
                if c['min'] is not None:
                    lower[pos] = max(lower[pos], float(c['min']))
                if c['max'] is not None:
                    upper[pos] = min(upper[pos], float(c['max']))

        return lower, upper

    def _apply_fixed_mean(self, theta, constraint, param_info, active_mask):
        relative = np.asarray(constraint['relative_indices'], dtype=np.int64)
        if relative.size == 0:
            return
        if np.any(relative < 0) or np.any(relative >= param_info['count']):
            raise ValueError(
                f"fixed_mean index out of bounds for {constraint['name']}"
            )

        absolute = param_info['start'] + relative
        values = np.asarray(theta[absolute], dtype=np.float64).copy()
        lower, upper = self._absolute_bounds(constraint['name'], absolute)
        target_sum = constraint['target_mean'] * len(values)

        if target_sum < np.sum(lower) - 1e-9 or target_sum > np.sum(upper) + 1e-9:
            raise ValueError(
                f"fixed_mean target is infeasible for {constraint['name']}"
            )

        adjustable = np.ones(len(values), dtype=bool)
        if active_mask is not None:
            adjustable = np.asarray(active_mask[absolute] != 0, dtype=bool)

        # Bounded water-filling projection. At each pass, share the residual
        # equally over still-adjustable entries, then retire entries that hit
        # a registered range bound.
        tolerance = 1e-10 * max(1.0, abs(target_sum))
        for _ in range(len(values) + 1):
            residual = target_sum - float(np.sum(values))
            if abs(residual) <= tolerance:
                break

            if residual > 0.0:
                free = adjustable & (values < upper - tolerance)
            else:
                free = adjustable & (values > lower + tolerance)
            free_count = int(np.count_nonzero(free))
            if free_count == 0:
                break

            values[free] += residual / free_count
            values[:] = np.minimum(np.maximum(values, lower), upper)

        residual = target_sum - float(np.sum(values))
        if abs(residual) > tolerance:
            raise ValueError(
                f"fixed_mean target cannot be reached by active entries of "
                f"{constraint['name']}"
            )

        if constraint.get('quantized', False):
            values = self._project_quantized_sum(
                values, lower, upper, adjustable, target_sum,
                constraint['name'], tolerance,
            )
        theta[absolute] = values

    @staticmethod
    def _project_quantized_sum(
        values, lower, upper, adjustable, target_sum, name, tolerance
    ):
        """Keep ``rint(values).sum()`` equal to an integer fixed-sum gauge.

        The evaluator and saved constants both use ``np.rint``. A continuous
        mean projection alone can therefore drift by one or more centipawns
        after quantization. Select the nearest feasible integer cells, then
        redistribute only the fractional residual so SPSA can still accumulate
        sub-centipawn updates.
        """
        target_int = int(np.rint(target_sum))
        if abs(target_sum - target_int) > tolerance:
            raise ValueError(
                f"quantized fixed_mean target must be integral for {name}"
            )

        quantized = np.rint(values).astype(np.int64)
        integer_residual = target_int - int(np.sum(quantized))
        direction = 1 if integer_residual > 0 else -1

        for _ in range(abs(integer_residual)):
            candidates = []
            for pos in np.flatnonzero(adjustable):
                proposed = quantized[pos] + direction
                # Stay strictly inside the target rint cell so half-to-even
                # ties cannot select the adjacent integer.
                cell_lo = np.nextafter(float(proposed) - 0.5, float(proposed))
                cell_hi = np.nextafter(float(proposed) + 0.5, float(proposed))
                feasible_lo = max(float(lower[pos]), cell_lo)
                feasible_hi = min(float(upper[pos]), cell_hi)
                if feasible_lo <= feasible_hi:
                    if values[pos] < feasible_lo:
                        cost = feasible_lo - values[pos]
                    elif values[pos] > feasible_hi:
                        cost = values[pos] - feasible_hi
                    else:
                        cost = 0.0
                    candidates.append((cost, int(pos)))
            if not candidates:
                raise ValueError(
                    f"quantized fixed_mean target cannot be reached for {name}"
                )
            _, chosen = min(candidates)
            quantized[chosen] += direction

        cell_lower = np.empty(len(values), dtype=np.float64)
        cell_upper = np.empty(len(values), dtype=np.float64)
        for pos, q in enumerate(quantized):
            cell_lower[pos] = max(
                float(lower[pos]), np.nextafter(float(q) - 0.5, float(q))
            )
            cell_upper[pos] = min(
                float(upper[pos]), np.nextafter(float(q) + 0.5, float(q))
            )
            if cell_lower[pos] > cell_upper[pos]:
                raise ValueError(
                    f"quantized fixed_mean cell is infeasible for {name}"
                )

        projected = np.minimum(np.maximum(values, cell_lower), cell_upper)
        for _ in range(len(projected) + 1):
            residual = target_sum - float(np.sum(projected))
            if abs(residual) <= tolerance:
                break
            if residual > 0.0:
                free = adjustable & (projected < cell_upper - tolerance)
            else:
                free = adjustable & (projected > cell_lower + tolerance)
            free_count = int(np.count_nonzero(free))
            if free_count == 0:
                break
            projected[free] += residual / free_count
            projected[:] = np.minimum(
                np.maximum(projected, cell_lower), cell_upper
            )

        if abs(target_sum - float(np.sum(projected))) > tolerance:
            raise ValueError(
                f"quantized fixed_mean float target cannot be reached for {name}"
            )
        if int(np.sum(np.rint(projected))) != target_int:
            raise ValueError(
                f"quantized fixed_mean integer target cannot be reached for {name}"
            )
        return projected

    def _apply_constraint(self, theta, constraint, active_mask=None):
        ctype = constraint['type']

        # --- Special case: range on absolute index ---
        if ctype == 'range_absolute':
            idx = constraint['abs_idx']
            min_val = constraint['min']
            max_val = constraint['max']
            if min_val is not None:
                theta[idx] = max(theta[idx], min_val)
            if max_val is not None:
                theta[idx] = min(theta[idx], max_val)
            return

        name = constraint['name']
        
        # Find parameter info
        param_info = None
        for p in self.pm.param_map:
            if p['name'] == name:
                param_info = p
                break
        
        if param_info is None:
            print(f"Warning: Constraint defined for unknown parameter '{name}'")
            return

        start = param_info['start']
        count = param_info['count']
        shape = param_info['shape']
        
        # Extract the slice (view)
        # Note: We work on the flat theta, but reshaping might be needed for logic
        param_slice = theta[start : start + count]

        if ctype == 'fixed_mean':
            self._apply_fixed_mean(theta, constraint, param_info, active_mask)

        elif ctype == 'range':
            min_val = constraint['min']
            max_val = constraint['max']
            
            if min_val is not None:
                param_slice[:] = np.maximum(param_slice, min_val)
            if max_val is not None:
                param_slice[:] = np.minimum(param_slice, max_val)

        elif ctype == 'monotonic':
            # Reshape to actual shape to handle axes
            # Reshaping a slice might trigger a copy if it's not contiguous, 
            # but usually slices of 1D arrays are contiguous unless strided weirdly.
            # To be safe, we assign back later.
            try:
                reshaped = param_slice.reshape(shape)
            except AttributeError:
                 # Should not happen if param_slice is numpy array
                 return 
            
            axis = constraint['axis']
            direction = constraint['direction']
            end_idx = constraint.get('end_idx', None)
            
            if len(shape) <= axis:
                 print(f"Warning: Axis {axis} out of bounds for shape {shape} of {name}")
                 return

            if end_idx is not None:
                # Apply monotonicity only on the slice [:end_idx] along axis
                if axis == 0:
                    sub = reshaped[:end_idx]
                    if direction == 'increasing':
                        np.maximum.accumulate(sub, axis=0, out=sub)
                    elif direction == 'decreasing':
                        np.minimum.accumulate(sub, axis=0, out=sub)
                    # The rest of the array (reshaped[end_idx:]) is not modified
                else:
                    # For axis != 0, generic slicing
                    idx = [slice(None)] * len(shape)
                    idx[axis] = slice(None, end_idx)
                    sub = reshaped[tuple(idx)]
                    if direction == 'increasing':
                        np.maximum.accumulate(sub, axis=axis, out=sub)
                    elif direction == 'decreasing':
                        np.minimum.accumulate(sub, axis=axis, out=sub)
            else:
                if direction == 'increasing':
                    # v[i] <= v[i+1]
                    # Enforce: v[i+1] = max(v[i+1], v[i])
                    # Using accumulate propagates the max value forward
                    np.maximum.accumulate(reshaped, axis=axis, out=reshaped)
                    
                elif direction == 'decreasing':
                    # v[i] >= v[i+1]
                    # Enforce: v[i+1] = min(v[i+1], v[i])
                    np.minimum.accumulate(reshaped, axis=axis, out=reshaped)
            
            # If reshaped was a view, param_slice is updated.
            # If reshaped was a copy, we must copy back.
            # Since we can't easily detect copy vs view without checking base/flags,
            # we just flatten and assign back to be 100% safe.
            param_slice[:] = reshaped.flatten()
