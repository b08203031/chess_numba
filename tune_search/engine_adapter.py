
import sys
import argparse
import importlib
import os

# Ensure the root directory is in the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

def patch_constants(args):
    """
    Patches the chess_engine.constants module with values from args.
    This must be done BEFORE importing any other engine modules that might JIT compile functions.
    """
    import chess_engine.classical.constants as constants
    
    # Iterate over all arguments and update corresponding constants
    for arg_name, arg_value in vars(args).items():
        # Only patch if it's a valid constant name (uppercase)
        # and if the value is not None
        if arg_name.isupper() and arg_value is not None:
            if hasattr(constants, arg_name):
                # We assume all params are integers for now, as most search params are.
                # If floats are needed, we might need type casting logic.
                # However, argparse type=int handles the input, so arg_value is already int.
                
                # Special handling for numpy types if needed, but python ints usually work fine
                # unless Numba is very strict about types matching the original definition.
                # Let's check the type of the original constant.
                original_value = getattr(constants, arg_name)
                
                if isinstance(original_value, int):
                    setattr(constants, arg_name, int(arg_value))
                elif isinstance(original_value, float):
                    setattr(constants, arg_name, float(arg_value))
                else:
                    # For safety, just set it. Numba might complain if type changes drastically.
                    setattr(constants, arg_name, arg_value)
                
                # print(f"Patched {arg_name}: {original_value} -> {arg_value}")
            else:
                print(f"Warning: Argument {arg_name} is not a valid constant in chess_engine.constants")

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Tuning Engine Wrapper")
    
    # Import config to know which arguments to expect
    from tune_search.param_config import SEARCH_PARAMS
    
    for param_name in SEARCH_PARAMS:
        # Add argument for each parameter. We use default=None to know if it was provided.
        parser.add_argument(f"--{param_name}", dest=param_name, type=int, default=None, help=f"Set {param_name}")
    
    # Allow passing through other arguments to main.py if needed, 
    # but strictly speaking, main.py reads from stdin (UCI).
    # We might need to handle args that main.py expects? 
    # main.py currently doesn't use argparse for configuration, it uses UCI.
    
    args, unknown = parser.parse_known_args()
    
    # 2. Patch Constants
    # Crucial: Import constants FIRST, patch them, THEN import everything else.
    try:
        patch_constants(args)
    except ImportError:
        # This might happen if requirements aren't met, but we are in the same env
        print("Error: Could not import chess_engine.classical.constants")
        sys.exit(1)

    # 3. Start Engine
    # We import main logic only AFTER patching.
    try:
        from main import uci_loop
        uci_loop()
    except ImportError as e:
        print(f"Error starting engine: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
