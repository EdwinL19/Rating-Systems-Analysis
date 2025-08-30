import numpy as np
import time
from func import prob_log
import math

pi = np.float32(np.pi)
pi2 = pi * pi

class Glicko2:
    # Initalizes simulation variables
    # Number of Players, Initial Ratings, Initial True Ratings, Initial RD, Initial Volatilities,
    # Volatility Constant
    def __init__(self, num_players, init_ratings, init_true_ratings=None, init_rating_devs=None, 
                 init_volatilities=None, volatility_const=0.6):
        if init_rating_devs is None:
            init_rating_devs = np.full(num_players, 350)
        if init_volatilities is None:
            init_volatilities = np.full(num_players, 0.06)
        self.ratings = np.array(init_ratings, dtype=np.float32)
        self.true_ratings = np.array(init_true_ratings, dtype=np.float32)
        self.rating_devs = np.array(init_rating_devs, dtype=np.float32)
        self.volatilities = np.array(init_volatilities, dtype=np.float32)
        self.num_players = num_players
        self.volatility_const = np.float32(volatility_const)

    # Add players with given initial data
    def add_players(self, init_ratings, init_true_ratings, init_rating_devs=None, init_volatilities=None):
        num_players = len(init_ratings)
        if init_rating_devs is None:
            init_rating_devs = np.full(num_players, 350)
        if init_volatilities is None:
            init_volatilities = np.full(num_players, 0.06)
        self.num_players += num_players
        self.ratings = np.append(self.ratings, init_ratings)
        self.true_ratings = np.append(self.true_ratings, init_true_ratings)
        self.rating_devs = np.append(self.rating_devs, init_rating_devs)
        self.volatilities = np.append(self.volatilities, init_volatilities)

    # Updates true ratings variable, id and true_rating can be arrays of any group of players
    def update_true_rating(self, id, true_rating):
        self.true_ratings[id] = np.asarray(true_rating)

    # Converts rating to Glicko2 scale
    def _convert_rating(self, r): 
        return (r - 1500) / 173.7178
    
    # Converts RD to Glicko2 scale
    def _convert_rd(self, rd):
        return rd / 173.7178
    
    # Reverts rating to standard scale
    def _revert_rating(self, r):
        return 173.7178 * r + 1500
    
    # Reverts RD to standard scale
    def _revert_rd(self, rd):
        return rd * 173.7178
    
    # Computes g, E, v, and delta values for update process
    def _g_E_v_delta(self, r, rj, rdj, outcomes):
        g = 1 / np.sqrt(1 + 3 * np.square(rdj) / pi2)
        E = 1 / (1 + np.exp(-g * (r - rj)))
        v = 1 / np.sum(np.square(g) * E * (1 - E))
        delta = v * np.sum(g * (outcomes - E))
        return g, E, v, delta
    
    # Computes f for update process
    def _f(self, x, delta, rd, v, a, tau):
        temp = (rd * rd) + v + np.exp(x)
        return (np.exp(x) * ((delta * delta) - (rd * rd) - v - np.exp(x)) / (2 * temp * temp)) - (x - a) / (tau * tau)
    
    # Computes new volatility for update process
    def _volatility_change(self, r, rd, IV, rj, rdj, outcomes):
        g, E, v, delta = self._g_E_v_delta(r, rj, rdj, outcomes)
        a = np.log(IV * IV)
        A = a
        eps, tau = 0.000001, self.volatility_const
        delta2, rd2 = delta * delta, rd * rd
        f = lambda x: self._f(x, delta, rd, v, a, tau)
        
        if (delta2 > rd2 + v):
            B = np.log(delta2 - rd2 - v)
        else:
            k = 1
            while(f(a - k * tau) < 0):
                k += 1
            B = a - k * tau

        f_A, f_B = f(A), f(B)

        while (np.abs(B - A) > eps):
            C = A + (A - B) * f_A / (f_B - f_A) 
            f_C = f(C)
            if (f_C * f_B <= 0):
                A, f_A = B, f_B
            else:
                f_A /= 2
            B, f_B = C, f_C

        new_IV = np.exp(A / 2)

        return new_IV, g, E, v
    
    # Computes new rating, RD, and volatility after matches with given opponents and outcomes
    # Only compatible with singular rating
    def update(self, rating, rd, IV, rating_opp, rd_opp, outcomes):
        rating, rating_opp = self._convert_rating(rating), self._convert_rating(rating_opp)
        rd, rd_opp = self._convert_rd(rd), self._convert_rd(rd_opp)

        new_IV, g, E, v = self._volatility_change(rating, rd, IV, rating_opp, rd_opp, outcomes)
        new_rd = 1 / np.sqrt(1 / ((rd * rd) + (new_IV * new_IV)) + 1 / v)
        new_rating = rating + (new_rd * new_rd) * np.sum(g * (outcomes - E))

        # print(v)

        new_rd = self._revert_rd(new_rd)
        new_rating = self._revert_rating(new_rating)

        return new_rating, new_rd, new_IV
    
    # Updates ratings for a match with id1 and id2 for given outcome
    # 1 = id1 wins, 0 = id2 wins
    def update_instant(self, id1, id2, outcome):
        t1, t2 = id1, id2
        r1 = self.ratings[t1]
        r2 = self.ratings[t2]
        rd1 = self.rating_devs[t1]
        rd2 = self.rating_devs[t2]
        vol1 = self.volatilities[t1]
        vol2 = self.volatilities[t2]

        self.ratings[t1], self.rating_devs[t1], self.volatilities[t1] = self.update(
            r1, rd1, vol1, r2, 
            rd2, outcome)
        self.ratings[t2], self.rating_devs[t2], self.volatilities[t2] = self.update(
            r2, rd2, vol2, r1, 
            rd1, not outcome)

    def update_period(self, id1, id2, outcomes):
        # have to copy player data and use the copy
        ratings = np.copy(self.ratings)
        rds = np.copy(self.rating_devs)
        vols = np.copy(self.volatilities)

        opp_data = [[] for _ in range(self.num_players)]
        for i in range(len(id1)):
            t1, t2, result = id1[i], id2[i], outcomes[i]
            opp_data[t1].append([t2, result])
            opp_data[t2].append([t1, not result])

        for i, data in enumerate(opp_data):
            if len(data) == 0:
                continue
            data = np.array(data)
            rating = ratings[i]
            RD = rds[i]
            vol = vols[i]
            id_opp = data[:, 0]
            outcome = data[:, 1]
            rating_opp = ratings[id_opp]
            rd_opp = rds[id_opp]

            self.ratings[i], self.rating_devs[i], self.volatilities[i] = self.update(
                rating, RD, vol, rating_opp, rd_opp, outcome)
            
    # Simulates matches for every player against opponents with the given ratings and RD
    # Default is opponents have the same rating and true rating
    # Default opponent is one with rating and true rating equal to the players' rating
    def match_all_perfect(self, n=1, rating_opp=None, true_rating_opp=None, rd_opp=None):
        if rating_opp is None:
            rating_opp = np.repeat(self.ratings.reshape(-1,1), n, axis=1)
        if rd_opp is None:
            rd_opp = np.zeros_like(rating_opp)
        if true_rating_opp is None:
            true_rating_opp = rating_opp
        win_prob = prob_log(self.true_ratings.reshape(-1,1), true_rating_opp)
        # print(win_prob)
        outcomes = np.random.rand(*np.shape(win_prob)) < win_prob
        #print(outcomes)

        for i in range(self.num_players):
            new_rating, new_rd, new_IV = self.update(self.ratings[i], self.rating_devs[i], 
                                                     self.volatilities[i], rating_opp[i], 
                                                     rd_opp[i], outcomes[i])
            self.ratings[i], self.rating_devs[i], self.volatilities[i] = new_rating, new_rd, new_IV
        
    # Simulates given number of random matches between all players initialized
    # Period is the number of games played
    def match_all_random(self, n=1):
        num_players = self.num_players
        matches = np.zeros((num_players, 2, n), dtype=int)

        for i in range(n):
            random = np.random.permutation(num_players)
            id1, id2 = random[::2], random[1::2]
            win_prob = prob_log(self.true_ratings[id1], self.true_ratings[id2])
            outcomes = np.random.rand(*np.shape(win_prob)) < win_prob
            matches[id1, 0, i], matches[id2, 0, i] = id2, id1
            matches[id1, 1, i], matches[id2, 1, i] = outcomes, np.logical_not(outcomes)

        r = np.copy(self.ratings)
        rd = np.copy(self.rating_devs)
        vol = np.copy(self.volatilities)
        for i in range(num_players):
            self.ratings[i], self.rating_devs[i], self.volatilities[i] = self.update(
                r[i], rd[i], vol[i], r[matches[i, 0]], 
                rd[matches[i, 0]], matches[i, 1])

    # Simulates given number of matches with rating based matchmaking
    # Players with similar ratings will be randomly paired into matches
    # Period is n, the number of games played
    def match_all_realistic(self, n=1, p=10):
        num_players = self.num_players
        matches = np.zeros((num_players, 2, n), dtype=int)

        if num_players < 100:
            p = 1

        sorted_rating_indices = np.argsort(self.ratings)
        # groups = np.array(np.array_split(sorted_rating_indices, p))
        # print(groups)

        
        
        for i in range(n):
            ratings = np.copy(self.ratings)
            noisy_ratings = ratings + np.random.normal(0, 30, size=len(ratings))
            ids = np.argsort(noisy_ratings)

            # rand_sorted_indices = np.copy(sorted_rating_indices)
            # for j in range(num_players):
            #     k = np.random.randint(max(0, j - p), min(num_players, j + p + 1))
            #     rand_sorted_indices[j], rand_sorted_indices[k] = rand_sorted_indices[k], rand_sorted_indices[j]

            # for group in groups:
            #     np.random.shuffle(group)
            # ids = np.array(groups).flatten()
 
            # print(self.ratings[ids])
            # print(self.true_ratings[rand_sorted_indices])
            
            # id1, id2 = rand_sorted_indices[::2], rand_sorted_indices[1::2]
            id1, id2 = ids[::2], ids[1::2]
            win_prob = prob_log(self.true_ratings[id1], self.true_ratings[id2])
            # print(win_prob)
            outcomes = np.random.rand(*np.shape(win_prob)) < win_prob
            # print(outcomes)
            matches[id1, 0, i], matches[id2, 0, i] = id2, id1
            matches[id1, 1, i], matches[id2, 1, i] = outcomes, np.logical_not(outcomes)

        r = np.copy(self.ratings)
        rd = np.copy(self.rating_devs)
        vol = np.copy(self.volatilities)
        for i in range(num_players):
            self.ratings[i], self.rating_devs[i], self.volatilities[i] = self.update(
                r[i], rd[i], vol[i], r[matches[i, 0]], 
                rd[matches[i, 0]], matches[i, 1])
            # if self.rating_devs[i] < 100:
            #     self.rating_devs[i] = 100
        # print(r[matches[0,0]])
        # print(matches)
            
    # Nicely formatted printing of all player data
    #  ID     Rating    True Rating    Rating Deviation   Volatilities
    #  ...      ...         ...              ...              ...
    def print_data(self):
        print(" ID     Rating    True Rating  Rating Deviation  Volatilities")
        for i in range(self.num_players):
            print(f"{i:<7}{self.ratings[i]:<11.2f}{self.true_ratings[i]:<13.2f}"
                  f"{self.rating_devs[i]:<11.2f}{self.volatilities[i]:11.6f}")

    def print_nba_data(self, names):
        indices = np.argsort(self.ratings)[::-1]
        ratings = self.ratings[indices]
        rd = self.rating_devs[indices]
        vol = self.volatilities[indices]
        sorted_names = names[indices]

        print("ID Name                     Rating     RD      Volatility")
        for i in range(self.num_players):
            print(f"{i:<3}{names[i]:<25}{self.ratings[i]:<11.2f}{self.rating_devs[i]:<10.2f}{self.volatilities[i]:<15.7f}{sorted_names[i]:<25}{ratings[i]:<11.2f}{rd[i]:<10.2f}{vol[i]:<15.7f}")

# start_time = time.time()

# num_players = 1000
# # np.random.seed(3213214)
# init_rating = np.full(num_players, 1500, dtype=np.float32)
# init_true_rating = np.random.normal(1500, 350, num_players).astype(np.float32)
# init_rd = np.full(num_players, 350, dtype=np.float32)
# init_vol = np.full(num_players, 0.06, dtype=np.float32)

# game = Glicko2(num_players, init_rating, init_rd, init_vol, 0.6, init_true_rating)

# # game.print_data()

# for _ in range(30):
#     game.match_all_realistic(10)

# game.print_data()

# print("--- %s seconds ---" % (time.time() - start_time))


    