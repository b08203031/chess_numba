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

    def apply(self, theta):
        """
        Applies all registered constraints to the theta vector in-place.
        """
        for c in self.constraints:
            self._apply_constraint(theta, c)

    def _apply_constraint(self, theta, constraint):
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

        if ctype == 'range':
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
