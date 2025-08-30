import numpy as np
#from scipy.stats import norm
#import time
from func import prob_log

# Implementation of simulation of the Elo rating system
class Elo:
    # Initalizes simulation variables
    def __init__(self, num_players, init_ratings, init_true_ratings=None, rating_const=50):
        self.ratings = np.array(init_ratings, dtype=np.float32)
        self.true_ratings = np.array(init_true_ratings, dtype=np.float32)
        self.num_players = num_players
        self.rating_const = rating_const
    
    # Add more players to simulation with given ratings and true ratings
    def add_players(self, init_ratings, init_true_ratings): 
        self.num_players += len(init_ratings)
        self.ratings = np.append(self.ratings, init_ratings)
        self.true_ratings = np.append(self.true_ratings, init_true_ratings)
    
    # Changes in rating after a 1v1 given outcome where 1 = r1 wins, 0 = r2 wins
    def rating_change(self, r1, r2, outcome):
        r1_win_prob = prob_log(r1, r2)
        change = self.rating_const * (outcome - r1_win_prob)
        return change
    
    def update(self, id1, id2, outcome):
        change = self.rating_change(self.ratings[id1], self.ratings[id2], outcome)
        self.ratings[id1] += change
        self.ratings[id2] -= change

    def batch_update(self, id1, id2, outcomes):
        for i in range(len(id1)):
            t1, t2 = id1[i], id2[i]
            change = self.rating_change(self.ratings[t1], self.ratings[t2], outcomes[i])
            self.ratings[t1] += change
            self.ratings[t2] -= change
    
    # Simulates matches for every player against opponents with the given ratings
    # Assume that opponents have the same rating and true rating
    # Default opponent is one with rating and true rating equal to the players' rating
    def match_all_t(self, ratings_opp=None, true_ratings_opp=None):
        if ratings_opp is None:
            ratings_opp = self.ratings
        if true_ratings_opp is None:
            true_ratings_opp = ratings_opp

        win_prob = prob_log(self.true_ratings, true_ratings_opp)
        #print(win_prob)
        outcomes = np.random.rand(self.num_players) < win_prob
        #print(outcomes)
        self.ratings += self.rating_change(self.ratings, ratings_opp, outcomes)

    # Nicely formatted printing of all player data
    #  ID     Rating    True Rating    
    #  ...      ...         ...          
    def print_data(self):
        print(" ID     Rating    True Rating")
        for i in range(self.num_players):
            print(f"{i:<7}{self.ratings[i]:<11.2f}{self.true_ratings[i]:<11.2f}")



# start_time = time.time()

# num_players = 2
# init_rating = np.full(num_players, 1500, dtype=np.float32)
# init_true_rating = np.random.normal(1500, 350, num_players).astype(np.float32)

# game = Elo(num_players, init_rating, init_true_rating)

# game.print_data()
# for _ in range(1):
#     rand_opp = np.random.normal(1500, 350, num_players).astype(np.float32)
#     #print(rand_opp)
#     game.match_all_t(rand_opp)
# game.print_data()

# print("--- %s seconds ---" % (time.time() - start_time))
