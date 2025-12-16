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

    def add_monotonic(self, param_name, axis=0, direction='increasing'):
        """
        Constrains a parameter array to be monotonic along a specific axis.
        
        Args:
            param_name (str): The name of the parameter.
            axis (int): The axis to enforce monotonicity on (0 for rows/first dim).
            direction (str): 'increasing' (v[i] <= v[i+1]) or 'decreasing' (v[i] >= v[i+1]).
        """
        self.constraints.append({
            'type': 'monotonic',
            'name': param_name,
            'axis': axis,
            'direction': direction
        })

    def apply(self, theta):
        """
        Applies all registered constraints to the theta vector in-place.
        """
        for c in self.constraints:
            self._apply_constraint(theta, c)

    def _apply_constraint(self, theta, constraint):
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

        if constraint['type'] == 'range':
            min_val = constraint['min']
            max_val = constraint['max']
            
            if min_val is not None:
                param_slice[:] = np.maximum(param_slice, min_val)
            if max_val is not None:
                param_slice[:] = np.minimum(param_slice, max_val)

        elif constraint['type'] == 'monotonic':
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
            
            if len(shape) <= axis:
                 print(f"Warning: Axis {axis} out of bounds for shape {shape} of {name}")
                 return

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
