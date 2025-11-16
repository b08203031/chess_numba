import numpy as np

def calculate_search_time(wtime_ms, btime_ms, winc_ms, binc_ms, movestogo, side_to_move):
    """
    Calculates the optimal and maximum search time for the current move.
    This is a simplified version of the logic used in engines like Stockfish.

    Args:
        wtime_ms: White's remaining time in milliseconds.
        btime_ms: Black's remaining time in milliseconds.
        winc_ms: White's increment in milliseconds.
        binc_ms: Black's increment in milliseconds.
        movestogo: The number of moves to the next time control.
        side_to_move: The side to move (0 for white, 1 for black).

    Returns:
        A dictionary containing 'optimum_time' and 'maximum_time' in milliseconds.
    """
    time_remaining_ms = wtime_ms if side_to_move == 0 else btime_ms
    increment_ms = winc_ms if side_to_move == 0 else binc_ms

    # If movestogo is not specified, estimate it. A common value is 30.
    if movestogo is None or movestogo == 0:
        movestogo = 30

    # Basic time allocation: a fraction of the remaining time plus a portion of the increment.
    # We use a safety margin by not using the full increment.
    optimum_time = (time_remaining_ms / movestogo) + (increment_ms * 0.8)

    # The maximum time is a multiple of the optimum time.
    # This gives the engine flexibility in critical positions.
    # However, it should not use up all the remaining time.
    maximum_time = optimum_time * 4

    # Ensure maximum_time does not exceed the remaining time, leaving a small buffer.
    safety_buffer_ms = 500  # Leave at least 0.5 seconds on the clock
    if maximum_time >= time_remaining_ms:
        maximum_time = time_remaining_ms - safety_buffer_ms
        if maximum_time < 0:
            maximum_time = 10 # Search for a very small amount of time if we're in deep trouble

    # Ensure optimum time is not greater than maximum time
    if optimum_time > maximum_time:
        optimum_time = maximum_time

    return {
        'optimum_time': int(optimum_time),
        'maximum_time': int(maximum_time)
    }
