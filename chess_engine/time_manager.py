import numpy as np

def calculate_search_time(wtime_ms, btime_ms, winc_ms, binc_ms, movestogo, side_to_move):
    """
    計算當前移動的最佳和最大搜尋時間。
    這是類似 Stockfish 等引擎中使用的邏輯的簡化版本。

    Args:
        wtime_ms (int): 白方剩餘時間（毫秒）。
        btime_ms (int): 黑方剩餘時間（毫秒）。
        winc_ms (int): 白方加秒（毫秒）。
        binc_ms (int): 黑方加秒（毫秒）。
        movestogo (int): 距離下一個時限的步數。如果沒有時限，則為 None。
        side_to_move (int): 行棋方（0 為白，1 為黑）。

    Returns:
        dict: 包含 'optimum_time' 和 'maximum_time' 的字典（毫秒）。
    """
    time_remaining_ms = wtime_ms if side_to_move == 0 else btime_ms
    increment_ms = winc_ms if side_to_move == 0 else binc_ms

    # If movestogo is not specified, estimate it. A common value is 30.
    # 如果未指定 movestogo，則進行估計。常用值為 30。
    if movestogo is None or movestogo == 0:
        movestogo = 30

    # Basic time allocation: a fraction of the remaining time plus a portion of the increment.
    # We use a safety margin by not using the full increment.
    # 基本時間分配：剩餘時間的一小部分加上一部分加秒。
    # 我們保留安全邊際，不使用全部加秒。
    optimum_time = (time_remaining_ms / movestogo) + (increment_ms * 0.8)

    # The maximum time is a multiple of the optimum time.
    # This gives the engine flexibility in critical positions.
    # However, it should not use up all the remaining time.
    # 最大時間是最佳時間的倍數。
    # 這給予引擎在關鍵局面下的靈活性。
    # 但是，它不應耗盡所有剩餘時間。
    maximum_time = optimum_time * 4

    # Ensure maximum_time does not exceed the remaining time, leaving a small buffer.
    # 確保最大時間不超過剩餘時間，並保留一個小的緩衝區。
    safety_buffer_ms = 500  # Leave at least 0.5 seconds on the clock / 至少保留 0.5 秒
    if maximum_time >= time_remaining_ms:
        maximum_time = time_remaining_ms - safety_buffer_ms
        if maximum_time < 0:
            maximum_time = 10 # Search for a very small amount of time if we're in deep trouble / 如果時間非常緊迫，只搜尋很短的時間

    # Ensure optimum time is not greater than maximum time
    # 確保最佳時間不大於最大時間
    if optimum_time > maximum_time:
        optimum_time = maximum_time

    return {
        'optimum_time': int(optimum_time),
        'maximum_time': int(maximum_time)
    }
