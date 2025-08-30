import numpy as np
import math
import random
import sys

LOG_BASE_10 = np.float32(math.log(10))

def old_prob_log(r1, r2):
        return 1.0 / (1 + 10 ** ((r2 - r1) / 400))

def prob_log(r1, r2):
    return 1.0 / (1 + np.exp((r2 - r1) / 400 * LOG_BASE_10))

def partial_argsort(arr, percent_sorted=1.0):
    assert 0 < percent_sorted <= 1.0, "percent_sorted must be between 0 and 1"

    n = len(arr)
    target_pivots = math.ceil(percent_sorted * n)
    placed_pivots = 0

    # Work on the index array instead of the original array
    indices = list(range(n))
    random.shuffle(indices)  # Helps avoid worst-case pivot behavior

    def partition(left, right):
        pivot_idx = random.randint(left, right)
        pivot_val = arr[indices[pivot_idx]]
        indices[pivot_idx], indices[right] = indices[right], indices[pivot_idx]

        i = left
        for j in range(left, right):
            if arr[indices[j]] < pivot_val:
                indices[i], indices[j] = indices[j], indices[i]
                i += 1

        indices[i], indices[right] = indices[right], indices[i]
        return i

    # Tail-recursion optimized quicksort
    stack = [(0, n - 1)]

    while stack and placed_pivots < target_pivots:
        left, right = stack.pop()

        while left < right and placed_pivots < target_pivots:
            pivot_index = partition(left, right)
            placed_pivots += 1

            # Push the larger subarray on the stack, recurse on smaller one
            if pivot_index - left < right - pivot_index:
                stack.append((pivot_index + 1, right))
                right = pivot_index - 1
            else:
                stack.append((left, pivot_index - 1))
                left = pivot_index + 1

    return indices

def probabilistic_matchmaking(ratings, scale=100.0):
    n = len(ratings)
    assert n % 2 == 0

    indices = np.arange(n)
    np.random.shuffle(indices)  # Randomize order

    matched = np.full(n, -1, dtype=int)
    used = np.zeros(n, dtype=bool)

    for i in indices:
        if used[i]:
            continue

        available = np.where(~used)[0]
        available = available[available != i]
        if len(available) == 0:
            break

        # Compute probability of matching based on similarity
        rating_diffs = np.abs(ratings[available] - ratings[i])
        probs = np.exp(-rating_diffs / scale)
        probs /= probs.sum()

        opponent = np.random.choice(available, p=probs)
        matched[i] = opponent
        matched[opponent] = i
        used[i] = used[opponent] = True

    # Extract final matches
    team_a = []
    team_b = []
    seen = set()
    for i, j in enumerate(matched):
        if j != -1 and (j, i) not in seen:
            team_a.append(i)
            team_b.append(j)
            seen.add((i, j))

    return np.array(team_a), np.array(team_b)