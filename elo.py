import numpy as np
import time
from func import prob_log
from func import probabilistic_matchmaking

# Implementation of simulation of the Elo rating system
class Elo:
    # Initalizes simulation variables
    # Number of Players, Initial Ratings, Inital True Ratings, Rating Change Constant
    def __init__(self, num_players, init_ratings, init_true_ratings=None, rating_const=32):
        self.ratings = np.array(init_ratings, dtype=np.float32)
        self.true_ratings = np.array(init_true_ratings, dtype=np.float32)
        self.num_players = num_players
        self.rating_const = rating_const

    def change_const(self, new_const):
        self.rating_const = new_const

    # Add more players to simulation with given initial ratings and true ratings
    def add_players(self, init_ratings, init_true_ratings): 
        self.num_players += len(init_ratings)
        self.ratings = np.append(self.ratings, init_ratings)
        self.true_ratings = np.append(self.true_ratings, init_true_ratings)

    # Updates true ratings variable, id and true_rating can be arrays of any group of players
    def update_true_rating(self, id, true_rating):
        self.true_ratings[id] = np.asarray(true_rating)
    
    # Changes in rating after a 1v1 given outcome
    # 1 = r1 wins, 0 = r2 wins
    def rating_change(self, r1, r2, outcome):
        r1_win_prob = prob_log(r1, r2)
        change = self.rating_const * (outcome - r1_win_prob)
        return change
    
    # Updates ratings for a match with id1 and id2 based on outcome
    # 1 = id1 wins, 0 = id2 wins
    def update(self, id1, id2, outcome):
        change = self.rating_change(self.ratings[id1], self.ratings[id2], outcome)
        self.ratings[id1] += change
        self.ratings[id2] -= change

    # Update ratings for consecutive games
    def batch_update(self, id1, id2, outcomes):
        for i in range(len(id1)):
            t1, t2 = id1[i], id2[i]
            change = self.rating_change(self.ratings[t1], self.ratings[t2], outcomes[i])
            self.ratings[t1] += change
            self.ratings[t2] -= change
    
    # Simulates matches for every player against opponents with the given ratings
    # Assume that opponents have the same rating and true rating
    # Default opponent is one with rating and true rating equal to the players' rating
    def match_all_perfect(self, ratings_opp=None, true_ratings_opp=None):
        if ratings_opp is None:
            ratings_opp = self.ratings
        if true_ratings_opp is None:
            true_ratings_opp = ratings_opp

        win_prob = prob_log(self.true_ratings, true_ratings_opp)
        #print(win_prob)
        outcomes = np.random.rand(self.num_players) < win_prob
        #print(outcomes)
        self.ratings += self.rating_change(self.ratings, ratings_opp, outcomes)

    # Simulates given number of random matches between all players initialized
    # Period is the number of games played
    def match_all_random(self):
        num_players = self.num_players

        random = np.random.permutation(num_players)
        id1, id2 = random[::2], random[1::2]

        # print(matches)
        win_prob = prob_log(self.true_ratings[id1], self.true_ratings[id2])
        # print(win_prob)
        # outcomes = np.zeros(self.num_players)
        # outcomes[::2] = np.random.rand(*np.shape(win_prob)) < win_prob
        # outcomes[1::2] = np.logical_not(outcomes[::2])
        outcomes = np.random.rand(*np.shape(win_prob)) < win_prob
        # print(outcomes)

        r = self.ratings
        change = self.rating_change(
            r[id1], r[id2], outcomes)
        self.ratings[id1] += change
        self.ratings[id2] -= change
    
    # Simulates given number of matches with rating based matchmaking
    # Players with similar ratings will be randomly paired into matches
    # Period is n, the number of games played
    def match_all_realistic(self, p=10):
        ratings = np.copy(self.ratings)
        # noisy_ratings = ratings + np.random.normal(0, 100, size=len(ratings))
        noisy_ratings = ratings + np.random.uniform(-50, 50, size=len(ratings))
        sorted_rating_indices = np.argsort(noisy_ratings)

        # groups = np.array(np.array_split(sorted_rating_indices, p))
        # for group in groups:
        #     np.random.shuffle(group)
        #     ids = np.array(groups).flatten()

        # for j in range(self.num_players):
        #     k = np.random.randint(max(0, j - p), min(self.num_players, j + p + 1))
        #     sorted_rating_indices[j], sorted_rating_indices[k] = sorted_rating_indices[k], sorted_rating_indices[j]
        # print(self.ratings[rand_sorted_indices])

        # id1, id2 = probabilistic_matchmaking(ratings)

        # id1, id2 = ids[::2], ids[1::2]
        id1, id2 = sorted_rating_indices[::2], sorted_rating_indices[1::2]
        # print("vs.")
        # index = np.where(sorted_rating_indices==0)[0]
        # if index % 2==0:
        #     index+=1
        # else:
        #     index-=1
        # print(self.ratings[sorted_rating_indices[index]])
        # print(self.true_ratings[sorted_rating_indices[index]])
        # print(matches)
        win_prob = prob_log(self.true_ratings[id1], self.true_ratings[id2])
        # sum = np.sum(win_prob < 0.3)
        # print(sum)
        outcomes = np.random.rand(*np.shape(win_prob)) < win_prob
        # print(outcomes)


        # # print(matches)
        # win_prob = prob_log(self.true_ratings.reshape(-1, 1), self.true_ratings[matches])
        # # print(win_prob)
        # outcomes = np.random.rand(*np.shape(win_prob)) < win_prob
        # # print(outcomes)

        r = self.ratings
        change = self.rating_change(r[id1], r[id2], outcomes)
        self.ratings[id1] += change
        self.ratings[id2] -= change

    # Nicely formatted printing of all player data
    #  ID     Rating    True Rating    
    #  ...      ...         ...          
    def print_data(self):
        print(" ID     Rating    True Rating")
        for i in range(self.num_players):
            print(f"{i:<7}{self.ratings[i]:<11.2f}{self.true_ratings[i]:<11.2f}")

    def print_nba_data(self, names):
        indices = np.argsort(self.ratings)[::-1]
        ratings = self.ratings[indices]
        sorted_names = names[indices]

        print("ID Name                     Rating")
        for i in range(self.num_players):
            print(f"{i:<3}{names[i]:<25}{self.ratings[i]:<11.2f}{sorted_names[i]:<25}{ratings[i]:<11.2f}")

# start_time = time.time()
# # np.random.seed(123)
# num_players = 10000
# init_rating = np.full(num_players, 1500, dtype=np.float32)
# init_true_rating = np.random.normal(1500, 350, num_players).astype(np.float32)

# game = Elo(num_players, init_rating, init_true_rating, 32)

# # game.print_data()
# for _ in range(400):
#     # rand_opp = np.random.normal(1500, 350, num_players).astype(np.float32)
#     # print(rand_opp)
#     game.match_all_realistic()
# game.print_data()

# print("--- %s seconds ---" % (time.time() - start_time))
