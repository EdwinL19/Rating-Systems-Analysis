import numpy as np
import time
from func import prob_log
import math

LOG_BASE_10 = np.float32(math.log(10))
pi = np.float32(np.pi)
pi2 = pi * pi

class Glicko:
    # Initalizes simulation variables
    # Number of Players, Initial Ratings, Initial True Ratings, Initial Rating Deviations
    def __init__(self, num_players, init_ratings, init_true_ratings=None, init_rating_devs=None):
        if init_rating_devs is None:
            init_rating_devs = np.full(num_players, 350)
        self.ratings = np.array(init_ratings, dtype=np.float32)
        self.true_ratings = np.array(init_true_ratings, dtype=np.float32)
        self.rating_devs = np.array(init_rating_devs, dtype=np.float32)
        self.num_players = num_players
        self._q = np.float32(math.log(10) / 400)
        self._q2 = self._q * self._q

    # Add players with given initial ratings and deviation
    def add_players(self, init_ratings, init_true_ratings, init_rating_devs=None):
        num_players = len(init_ratings)
        if init_rating_devs is None:
            init_rating_devs = np.full(num_players, 350)
        self.num_players += num_players
        self.ratings = np.append(self.ratings, init_ratings)
        self.true_ratings = np.append(self.true_ratings, init_true_ratings)
        self.rating_devs = np.append(self.rating_devs, init_rating_devs)
    
    # Updates true ratings variable, id and true_rating can be arrays of any group of players
    def update_true_rating(self, id, true_rating):
        self.true_ratings[id] = np.asarray(true_rating)

    # Computes g and E values for update process
    def _g_E(self, r, rj, rdj):
        g = 1 / np.sqrt(1 + 3 * (self._q2) * np.square(rdj) / pi2)
        E = 1 / (1 + np.exp(-g * (r - rj) / 400 * LOG_BASE_10))
        return g, E

    # Computes new rating and deviation for matches with given outcome and opponent data
    # Compatible with all logical argument types
    # rating = [p1,p2,...]    rd = [p1,p2,...]  rating_opp = [[p1_1,p1_2,...],[p2_1,p2_2,...],...]
    # rd_opp = [[p1_1,p1_2,...],[p2_1,p2_2,...],...]    outcomes = [[1,1,...],[0,1,...],...]
    def update(self, rating, rd, rating_opp, rd_opp, outcomes):
        rating = np.asarray(rating)
        rating_opp = np.asarray(rating_opp)
        g, E = self._g_E(rating if rating.ndim != 1 else rating.reshape(-1,1), 
                         rating_opp, rd_opp)
        d2 = 1 / ((self._q2) * np.sum(np.square(g) * E * (1 - E), 
                                          axis=1 if rating_opp.ndim == 2 else None))
        # print(g)
        # print(E)
        # print(d2)
        new_rd = 1 / (1 / np.square(rd) + 1 / d2)
        new_rating = rating + (self._q * new_rd * np.sum(
            g * (outcomes - E), axis=1 if rating_opp.ndim == 2 else None))
        new_rd = np.sqrt(new_rd)
        
        new_rd = np.maximum(new_rd, 30)
        return new_rating, new_rd
    
    # Updates ratings for a match with id1 and id2 for given outcome
    # 1 = id1 wins, 0 = id2 wins
    def update_instant(self, id1, id2, outcome):
        t1, t2 = id1, id2
        r1 = self.ratings[t1]
        r2 = self.ratings[t2]
        rd1 = self.rating_devs[t1]
        rd2 = self.rating_devs[t2]

        self.ratings[t1], self.rating_devs[t1] = self.update(
            r1, rd1, r2, 
            rd2, outcome)
        self.ratings[t2], self.rating_devs[t2] = self.update(
            r2, rd2, r1, 
            rd1, not outcome)

    def update_period(self, id1, id2, outcomes):
        ratings = np.copy(self.ratings)
        rds = np.copy(self.rating_devs)

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
            id_opp = data[:, 0]
            outcome = data[:, 1]
            rating_opp = ratings[id_opp]
            rd_opp = rds[id_opp]

            self.ratings[i], self.rating_devs[i] = self.update(
                rating, RD, rating_opp, rd_opp, outcome)
            
    # Update ratings for consecutive games
    def batch_update_instant(self, id, id_opp, outcomes):
        for i in range(len(id)):
            t1, t2 = id[i], id_opp[i]
            self.ratings[t1], self.rating_devs[t1] = self.update(self.ratings[t1], self.rating_devs[t1], self.ratings[t2],
                                 self.rating_devs[t2], outcomes[i])
            self.ratings[t2], self.rating_devs[t2] = self.update(self.ratings[t2], self.rating_devs[t2], self.ratings[t1],
                                 self.rating_devs[t1], not outcomes[i])

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
        outcomes = np.random.rand(*np.shape(win_prob)) < win_prob
        # print(win_prob)
        # print(outcomes)
        self.ratings, self.rating_devs = self.update(self.ratings, self.rating_devs, rating_opp, rd_opp, outcomes)

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
        # for i in range(num_players):
        #     self.ratings[i], self.rating_devs[i] = self.update(
        #         r[i], rd[i], r[matches[i, 0]], 
        #         rd[matches[i, 0]], matches[i, 1])

        # print(r[matches[:, 0]])
        self.ratings, self.rating_devs = self.update(
                r, rd, r[matches[:, 0]], 
                rd[matches[:, 0]], matches[:, 1])
    
    # Simulates given number of matches with rating based matchmaking
    # Players with similar ratings will be randomly paired into matches
    # Period is n, the number of games played
    def match_all_realistic(self, n=1, p=10):
        num_players = self.num_players
        matches = np.zeros((num_players, 2, n), dtype=int)
        
        # sorted_rating_indices = np.argsort(self.ratings)

        for i in range(n):
            ratings = np.copy(self.ratings)
            # rand_sorted_indices = np.copy(sorted_rating_indices)
            noisy_ratings = ratings + np.random.normal(0, 30, size=len(ratings))
            sorted_rating_indices = np.argsort(noisy_ratings)

            # for j in range(num_players):
            #     k = np.random.randint(max(0, j - p), min(num_players, j + p + 1))
            #     rand_sorted_indices[j], rand_sorted_indices[k] = rand_sorted_indices[k], rand_sorted_indices[j]
            # # print(self.ratings[rand_sorted_indices])
            # rand_sorted_indices = rand_sorted_indices.reshape(-1, 2)
            # for id1, id2 in rand_sorted_indices:
            #     matches[id1, i], matches[id2, i] = id2, id1
            
            # for j in range(num_players):
            #     k = np.random.randint(max(0, j - p), min(num_players, j + p + 1))
            #     rand_sorted_indices[j], rand_sorted_indices[k] = rand_sorted_indices[k], rand_sorted_indices[j]
            
            # print(self.ratings[rand_sorted_indices])
            # print(self.true_ratings[rand_sorted_indices])
            # id1, id2 = sorted_rating_indices[::2], sorted_rating_indices[1::2]

            id1, id2 = sorted_rating_indices[::2], sorted_rating_indices[1::2]
            win_prob = prob_log(self.true_ratings[id1], self.true_ratings[id2])
            # print(win_prob)
            outcomes = np.random.rand(*np.shape(win_prob)) < win_prob
            # print(outcomes)
            matches[id1, 0, i], matches[id2, 0, i] = id2, id1
            matches[id1, 1, i], matches[id2, 1, i] = outcomes, np.logical_not(outcomes)

        # # print(matches)
        # win_prob = prob_log(self.true_ratings.reshape(-1, 1), self.true_ratings[matches])
        # # print(win_prob)
        # outcomes = np.random.rand(*np.shape(win_prob)) < win_prob
        # # print(outcomes)

        r = np.copy(self.ratings)
        rd = np.copy(self.rating_devs)
        # for i in range(num_players):
        #     self.ratings[i], self.rating_devs[i] = self.update(
        #         r[i], rd[i], r[matches[i]], 
        #         rd[matches[i]], outcomes[i])
        
        self.ratings, self.rating_devs = self.update(
                r, rd, r[matches[:, 0]], 
                rd[matches[:, 0]], matches[:, 1])
            
    # Nicely formatted printing of all player data
    #  ID     Rating    True Rating    Rating Deviation
    #  ...      ...         ...              ...
    def print_data(self):
        print(" ID     Rating    True Rating  Rating Deviation")
        for i in range(self.num_players):
            print(f"{i:<7}{self.ratings[i]:<11.2f}{self.true_ratings[i]:<13.2f}{self.rating_devs[i]:<11.2f}")

    def print_nba_data(self, names):
        indices = np.argsort(self.ratings)[::-1]
        ratings = self.ratings[indices]
        rd = self.rating_devs[indices]
        sorted_names = names[indices]

        print("ID Name                     Rating    RD")
        for i in range(self.num_players):
            print(f"{i:<3}{names[i]:<25}{self.ratings[i]:<11.2f}{self.rating_devs[i]:<10.2f}{sorted_names[i]:<25}{ratings[i]:<11.2f}{rd[i]:<10.2f}")



# start_time = time.time()
# np.random.seed(123)
# num_players = 1000
# init_rating = np.full(num_players, 1500, dtype=np.float32)
# init_true_rating = np.random.normal(1500, 350, num_players).astype(np.float32)
# init_rd = np.full(num_players, 350, dtype=np.float32)
# init_vol = np.full(num_players, 0.06, dtype=np.float32)

# game = Glicko(num_players, init_rating, init_true_rating, init_rd)

# # game.print_data()

# for _ in range(60):
#     # and_opp = np.full((num_players, 1), 1500, dtype=np.float32)
#     game.match_all_random(5)

# game.print_data()

# print("--- %s seconds ---" % (time.time() - start_time))