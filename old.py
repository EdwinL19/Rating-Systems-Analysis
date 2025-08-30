import numpy as np
from scipy.stats import norm
import time
import math
import matplotlib.pyplot as plt

# Variables
num_players = 10000
num_matches = 500
init_rating = 2000
norm_str_dist = [3500, 1500] # [SD, Mean]
win_prob_constant = 400 # Weighting constant for win probability funcvtion (400 is used in Elo)
win_prob_func = 'log_win_probability'
rating_const = 50

# Initializing players and player data
def initPlayerData():
    player_data = np.zeros((num_players, 3))  # [ID, Rating, Strength]
    player_data[:, 0] = np.arange(0, num_players)  # IDs
    player_data[:, 1] = init_rating  # Initial Rating
    player_data[:, 2] = np.random.normal(norm_str_dist[0], norm_str_dist[1], num_players) # Strength, N(3500, 1500^2)
    return player_data

# Matchmaking system to pair players into matches
def matchmaking(player_data):
    # Select players to add to queue with a 100% chance
    inQueue = player_data[np.random.random(player_data.shape[0]) < 2]
    
    # If odd number of players, remove a random player
    if len(inQueue) % 2 == 1:
        inQueue = np.delete(inQueue, np.random.randint(0, len(inQueue)-1), axis=0)

    # Sort by rating
    sorted_queue = inQueue[inQueue[:, 1].argsort()]
    
    # Create matches by pairing consecutive players
    return sorted_queue.reshape(-1, 2, 4)  # Each match has two players with 4 data points

# Win Probability Function based on Normal distribution
# Given player 1 and player 2's ratings, computes the probability player 1 wins
def norm_win_prob(rating1, rating2):
    rating_diffs = rating1 - rating2
    std_devs = rating_diffs*abs(rating_diffs) / 40000
    win_probs = norm.cdf(std_devs)
    
    # Clip values between 0 and 1
    win_probs = np.clip(win_probs, 0, 1)
    return win_probs

# Win Probability Function based on Logistic base 10
# Given player 1 and player 2's ratings, computes the probability player 1 wins
def log_win_probability(rating1, rating2):
    return 1.0 / (1 + 10 ** ((rating2 - rating1) / win_prob_constant))

# Computes elo change for player 1 given real outcome and chosen win probability function
# outcome = 1 means P1 won, outcome = 0 means P1 lost
def elo_rating(rating1, rating2, K, outcome):
    func = globals().get(win_prob_func)
    P = func(rating1, rating2)
    rating1 = rating1 + K * (outcome - P)
    return rating1

# Function to adjust ratings after each match
def elo_adjustment(player_data, win_percents):
    for i,player in enumerate(player_data):
        Ra = Rb = player[1]  # ratings
        P1_win, P2_win = elo_rating(Ra, Rb, rating_const, 1), elo_rating(Ra, Rb, rating_const, 0)
        
        # Update ratings based on win probabilities
        if np.random.random() < win_percents[i]:
            player_data[i, 1] = P1_win
        else:
            player_data[i, 1] = P2_win

# Placement matches rating adjustment
def unranked_elo_adjustment(player_data, win_percents):
    for i,player in enumerate(player_data):
        Ra = Rb = player[1]  # Skill ratings
        P1_win, P2_win = elo_rating(Ra, Rb, rating_const*4, 1), elo_rating(Ra, Rb, rating_const*4, 0)
        
        # Update ratings based on win probabilities
        if np.random.random() < win_percents[i]:
            player_data[i, 1] = P1_win
        else:
            player_data[i, 1] = P2_win

def sim_match_all(inPlacement):
    # Get real win probabilities
    win_percents = globals().get(win_prob_func)(player_data[:, 2], player_data[:, 1])

    # Check if in placement matches
    if(inPlacement):
        unranked_elo_adjustment(player_data, win_percents)
    else:
        elo_adjustment(player_data, win_percents)



# Running simulation
start_time = time.time()

# Initialize player data
player_data = initPlayerData()

# Visualization stuff
y = np.array([np.sum(abs(player_data[:,2] - player_data[:,1]))])
rand_player = np.random.randint(0,num_players)
y2 = np.array([player_data[rand_player, 1]])

# Simulation of num_matches iterations
for i in range(num_matches):  
    if(i<20):
        sim_match_all(1)
    else:
        sim_match_all(0)

    # Visualization stuff
    y = np.append(y, np.sum(abs(player_data[:,2] - player_data[:,1])))
    y2 = np.append(y2, player_data[rand_player, 1])

# Printing the result
x = np.arange(num_matches+1)
np.set_printoptions(
    formatter={
        'float_kind': lambda x: f"{x:.2f}",
        'int': lambda x: f"{x}"
    }
)
print(" ID     sRating    tRating    Volatility")
for row in player_data:
    print(f"{int(row[0]):<7}{row[1]:<11.2f}{row[2]:<11.2f}")
print("--- %s seconds ---" % (time.time() - start_time))

# Visualization
fig, (ax1,ax2,ax3) = plt.subplots(3,1)
ax1.plot(x,y,label = 'Total Difference between Rating and True Rating')
ax1.set_xlabel('Games Played')
ax1.set_ylabel('Total Difference')
ax1.grid(True)
ax1.legend()

ax2.plot(x[100:320],y[100:320],label = 'Total Difference between Rating and True Rating')
ax2.set_xlabel('Games Played')
ax2.set_ylabel('Total Difference')
ax2.grid(True)
ax2.legend()

ax3.plot(x,y2, label = 'Rating')
ax3.plot(x,[player_data[rand_player, 2]]*len(x), label = 'Strength (True Rating)')
ax3.set_xlabel('Games Played')
ax3.set_ylabel('Rating')
ax3.grid(True)
ax3.legend()

plt.show()