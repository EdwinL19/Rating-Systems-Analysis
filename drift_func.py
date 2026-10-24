import numpy as np
import math
from func import prob_log

# Logistic slope of the Elo win probability: prob_log(r1, r2) = 1 / (1 + exp(-C * (r1 - r2)))
C = math.log(10) / 400

# Simulates Elo for several independent configurations at once (one configuration per row)
# K, h:        rating change constant and matchmaking noise half-width for each row
# match_on:    "rating" (skill based matchmaking on visible ratings), "true" (oracle matchmaking on true ratings)
# rho:         per-row probability that a round uses random matchmaking instead
# bounded:     optional (low, high) clip applied after each round, like Elo.match_all_realistic(bounded=True)
# k_schedule:  optional function of the round number t returning a multiplier for K (scalar or one per row)
# keep_ratings: also store a copy of all ratings at every log point (key "history", shape (logs, R, N))
# init_ratings: starting visible ratings (default: everyone at 1500)
# Returns time series logged every log_every rounds and the final ratings
def simulate(K, h, num_players=1000, num_rounds=50000, true_ratings=None, match_on="rating",
             rho=0.0, bounded=None, k_schedule=None, keep_ratings=False, init_ratings=None,
             log_every=1000, seed=0):
    rng = np.random.default_rng(seed)
    match_on = np.atleast_1d(np.asarray(match_on))
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    K = np.atleast_1d(np.asarray(K, dtype=float))
    h = np.atleast_1d(np.asarray(h, dtype=float))
    R = max(len(K), len(h), len(match_on), len(rho))
    K, h = np.broadcast_to(K, (R,))[:, None], np.broadcast_to(h, (R,))[:, None]
    rho, match_on = np.broadcast_to(rho, (R,))[:, None], np.broadcast_to(match_on, (R,))[:, None]
    base_K = K

    if true_ratings is None:
        true_ratings = rng.normal(1500, 350, (R, num_players))
    theta = np.broadcast_to(np.asarray(true_ratings, dtype=float), (R, num_players)).copy()
    if init_ratings is None:
        init_ratings = 1500.0
    ratings = np.broadcast_to(np.asarray(init_ratings, dtype=float), (R, num_players)).copy()
    rows = np.arange(R)[:, None]

    log = {"t": [], "mse": [], "slope": [], "spearman": [], "matched_de2": [], "matched_dr2": [],
           "injection": [], "dissipation": []}
    if keep_ratings:
        log["history"] = []
    injection, dissipation = np.zeros(R), np.zeros(R)
    de2_sum, dr2_sum, window_count = np.zeros(R), np.zeros(R), 0

    for t in range(1, num_rounds + 1):
        if k_schedule is not None:
            K = base_K * np.broadcast_to(np.asarray(k_schedule(t), dtype=float), (R,))[:, None]

        # Matchmaking: sort by a noisy key and pair neighbours, same as Elo.match_all_realistic
        key = np.where(match_on == "true", theta, ratings) + rng.uniform(-1, 1, (R, num_players)) * h
        use_random = rng.random((R, 1)) < rho
        key = np.where(use_random, rng.random((R, num_players)), key)
        order = np.argsort(key, axis=1)
        id1, id2 = order[:, ::2], order[:, 1::2]

        p = prob_log(theta[rows, id1], theta[rows, id2])      # true win probability
        q = prob_log(ratings[rows, id1], ratings[rows, id2])  # predicted win probability
        dr = ratings[rows, id1] - ratings[rows, id2]
        de = dr - (theta[rows, id1] - theta[rows, id2])

        # Exact expected change of sum(e^2) this round, split into noise injection and dissipation
        injection += (2 * K**2 * (p * (1 - p) + (p - q) ** 2)).sum(axis=1) / num_players
        dissipation += (2 * K * de * (q - p)).sum(axis=1) / num_players
        de2_sum += (de**2).mean(axis=1)
        dr2_sum += (dr**2).mean(axis=1)
        window_count += 1

        outcomes = rng.random(p.shape) < p
        change = K * (outcomes - q)
        ratings[rows, id1] += change
        ratings[rows, id2] -= change
        if bounded is not None:
            np.clip(ratings, bounded[0], bounded[1], out=ratings)

        if t % log_every == 0:
            log["t"].append(t)
            log["mse"].append(centered_mse(ratings, theta))
            log["slope"].append(stretch(ratings, theta))
            log["spearman"].append(spearman(ratings, theta))
            log["matched_de2"].append(de2_sum / window_count)
            log["matched_dr2"].append(dr2_sum / window_count)
            log["injection"].append(injection.copy())
            log["dissipation"].append(dissipation.copy())
            if keep_ratings:
                log["history"].append(ratings.copy())
            de2_sum, dr2_sum, window_count = np.zeros(R), np.zeros(R), 0

    out = {k: np.array(v) for k, v in log.items()}
    out["ratings"], out["true_ratings"] = ratings, theta
    return out

# Mean squared rating error after removing the constant offset (Elo only identifies ratings up to a shift)
def centered_mse(ratings, theta):
    e = ratings - theta
    e = e - e.mean(axis=-1, keepdims=True)
    return np.mean(e**2, axis=-1)

# Regression slope of visible rating on true rating; 1 means the rating scale is calibrated
def stretch(ratings, theta):
    tc = theta - theta.mean(axis=-1, keepdims=True)
    rc = ratings - ratings.mean(axis=-1, keepdims=True)
    # Undefined (nan) when all true ratings are equal
    with np.errstate(invalid="ignore", divide="ignore"):
        return (tc * rc).sum(axis=-1) / (tc**2).sum(axis=-1)

# Spearman rank correlation between visible and true ratings, row by row
def spearman(ratings, theta):
    n = ratings.shape[-1]
    rank_r = np.argsort(np.argsort(ratings, axis=-1), axis=-1)
    rank_t = np.argsort(np.argsort(theta, axis=-1), axis=-1)
    return 1 - 6 * np.sum((rank_r - rank_t) ** 2, axis=-1) / (n * (n**2 - 1))

# Expected excess log loss (KL from true to predicted win probability) over the given pairs
def excess_log_loss(ratings, theta, id1, id2):
    p = prob_log(theta[id1], theta[id2])
    q = np.clip(prob_log(ratings[id1], ratings[id2]), 1e-12, 1 - 1e-12)
    return np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))

# Pairs produced by one round of skill based matchmaking with noise half-width h
def matched_pairs(ratings, h=50, rng=np.random):
    order = np.argsort(ratings + rng.uniform(-h, h, len(ratings)))
    return order[::2], order[1::2]

# Uniformly random pairs (with replacement) for held-out evaluation
def random_pairs(num_players, num_pairs=200000, rng=np.random):
    id1, id2 = rng.randint(0, num_players, num_pairs), rng.randint(0, num_players, num_pairs)
    keep = id1 != id2
    return id1[keep], id2[keep]

# Critical squared opponent gap: drift is unbounded when E[(r_i - r_j)^2] between opponents is below this
def critical_gap2(K):
    return K / C

# Predicted equilibrium stretch s* = W / (W - K/c) for mean squared opponent rating gap W (inf if W <= K/c)
def predicted_stretch(W, K):
    W = np.asarray(W, dtype=float)
    crit = critical_gap2(K)
    return np.where(W > crit, W / np.maximum(W - crit, 1e-12), np.inf)

# Effective diffusion coefficient of the rating density once games are coin flips: D = (K/8)(K - cW)
# Positive (the population spreads) exactly when the opponent gap W is below the critical value K/c
def effective_diffusion(K, W):
    return K / 8 * (K - C * np.asarray(W, dtype=float))

# Amplitudes mean(cos(k pi r / L)) of the cosine modes of the rating distribution on [0, length]
def cosine_modes(ratings, length, k):
    r = np.asarray(ratings, dtype=float).ravel()
    return np.array([np.mean(np.cos(kk * np.pi * r / length)) for kk in np.atleast_1d(k)])

# Rating density on [0, length] after dt rounds predicted by the heat equation with zero-flux walls,
# started from the empirical distribution of the ratings r0
def heat_density(r0, D, dt, x, length, kmax=100):
    k = np.arange(1, kmax + 1)
    a = 2 / length * cosine_modes(r0, length, k)
    decay = np.exp(-D * (k * np.pi / length) ** 2 * dt)
    return 1 / length + (a * decay) @ np.cos(np.outer(k, np.asarray(x, dtype=float)) * np.pi / length)

# Predicted SD of a player's rating around their percentile position, sqrt(s K / 2c),
# where s is the local stretch (rating points per true rating point)
def fluctuation_sd(local_stretch, K):
    return np.sqrt(np.asarray(local_stretch, dtype=float) * K / (2 * C))
