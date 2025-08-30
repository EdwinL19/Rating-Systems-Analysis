import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import rankdata

# Runs the simulation for num_games games and collects data
def run_simulation(elo, num_games, num_players, k_len, rank_period, model_name, match_type, rating_period=10):
    if match_type == "perfect":
        match = [elo[i].match_all_perfect for i in range(k_len)]
    elif match_type == "realistic":
        match = [elo[i].match_all_realistic for i in range(k_len)]
    elif match_type == "random":
        match = [elo[i].match_all_random for i in range(k_len)]
    
        
    # Average absolute difference between true and visible ratings over time
    avg_abs_diff_elo = [[np.mean(np.abs(elo[i].true_ratings - elo[i].ratings))] for i in range(k_len)]
    avg_sq_diff_elo = [[np.mean(np.square(elo[i].true_ratings - elo[i].ratings))] for i in range(k_len)]

    elo_ranking_dev = np.zeros(k_len)
    adj_period_ratings = [[np.copy(elo[i].ratings)] for i in range(k_len)]
    # print(adj_period_ratings)

    for i in range(k_len):
        # Ranking players by true rating for rank deviation data
        true_ranks = rankdata(elo[i].true_ratings, method='ordinal')                                         
        for j in range(num_games):
            if model_name == "elo":
                match[i]()
            elif model_name == "glicko" or model_name == "glicko2":
                if (j+1) % (i+1) == 0:
                    match[i](i+1) 
            elif model_name == "glicko2_vol":
                if (j+1) % rating_period == 0:
                    match[i](rating_period)

            # Tracking avg absolute rating deviation from true ratings
            avg_abs_diff_elo[i].append(np.mean(np.abs(elo[i].true_ratings - elo[i].ratings)))
            avg_sq_diff_elo[i].append(np.mean(np.square(elo[i].true_ratings - elo[i].ratings)))
            
            if (j+1) % rank_period == 0:
                # Ranking players by rating
                visible_ranks = rankdata(elo[i].ratings, method='ordinal')

                # Compute average ranking deviation
                ranking_deviation = np.sum(np.abs(visible_ranks - true_ranks) / (num_players - 1)) * 100 # Normalize to percentage
                elo_ranking_dev[i] += ranking_deviation

            if j < 30:
                adj_period_ratings[i].append(np.copy(elo[i].ratings))
                    
    elo_ranking_dev /= num_players * (num_games / rank_period)

    avg_abs_diff_elo, avg_sq_diff_elo = np.array(avg_abs_diff_elo), np.array(avg_sq_diff_elo)
    elo_ranking_dev, adj_period_ratings = np.array(elo_ranking_dev), np.array(adj_period_ratings)
    return avg_abs_diff_elo, avg_sq_diff_elo, elo_ranking_dev, adj_period_ratings

def avg_games_to_threshold(avg_abs_diff_elo):
    # Average number of games for player to be within 25, 50, 75, 100 rating from their true rating
    elo_avg_conv_25, elo_avg_conv_50 = np.argmax(avg_abs_diff_elo <= 25, axis=1), np.argmax(avg_abs_diff_elo <= 50, axis=1)
    elo_avg_conv_75, elo_avg_conv_100 = np.argmax(avg_abs_diff_elo <= 75, axis=1), np.argmax(avg_abs_diff_elo <= 100, axis=1)

    return elo_avg_conv_25, elo_avg_conv_50, elo_avg_conv_75, elo_avg_conv_100

# This function finds the point where the average rating difference from true rating starts to flatten out
def find_convergence_start(arr, window=100, slope_thresh=0.001):
    arr = np.asarray(arr)
    n = len(arr)
    
    for i in range(n - window):
        # Fit a linear trend over the window
        y = arr[i:i+window]
        x = np.arange(window)
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        if abs(slope) < slope_thresh:
            low = np.min(y)
            high = np.max(y)
            return i, coeffs, low, high  # First point where it starts flattening, coeffs[1] is the convergence value
    
    return -1, None, None, None  # If no flat region is found

def conv_stats_mse(k_len, num_games, avg_abs_diff_elo):
    elo_conv_stats = []
    elo_conv_mse = []
    for i in range(k_len):
        # Aggregate rating deviation convergence statistics
        # Slope and intercept will be used for volatility analysis
        conv_start, conv_coeffs, conv_low, conv_high = find_convergence_start(avg_abs_diff_elo[i], 100, 1e-3)
        conv_avg = conv_coeffs[1] if conv_coeffs is not None else None
        elo_conv_stats.append((conv_start, conv_avg, conv_low, conv_high))
        
        # Volatility Analysis based on mean squared error
        if conv_start != -1:
            conv_x = np.arange(conv_start, num_games+1)
            conv_y = avg_abs_diff_elo[i, conv_start:]

            conv_pred = np.polyval(conv_coeffs, conv_x)
        
            conv_residuals = conv_y - conv_pred
        
            conv_ssr = np.sum(np.square(conv_residuals))
        
            conv_mse = conv_ssr / (num_games - conv_start - 2) # -2 to account for linear regression

            elo_conv_mse.append(conv_mse)
        else:
            elo_conv_mse.append(-1)
            
    return elo_conv_stats, elo_conv_mse

def adj_mse(k_len, num_players, adj_period_ratings):
    elo_adj_mse = []
    player_data = adj_period_ratings.transpose(0, 2, 1)
    adj_x = np.arange(0, 31) # To include the 30th game played
    for i in range(k_len):
        sum_mse = 0
        for j in range(num_players):
            adj_y = player_data[i, j, :]
            adj_coeffs = np.polyfit(adj_x, adj_y, 1)
            adj_pred = np.polyval(adj_coeffs, adj_x)
            adj_residuals = adj_y - adj_pred
            adj_ssr = np.sum(np.square(adj_residuals))
            adj_mse = adj_ssr / (31 - 2)
            sum_mse += adj_mse
        avg_mse = sum_mse / num_players
        elo_adj_mse.append(avg_mse)

    return elo_adj_mse

# Analysis of Absolute and Squared Rating Deviation Over Time
def graph_aggregate_avg_diffs(num_games, k_values, avg_abs_diff_elo, avg_sq_diff_elo):
    
    # Get length of k_range for looping
    k_len = len(k_values)
    # Create x-axis for number of games
    x_axis = np.arange(num_games + 1)  # +1 because we have initial state (0 games played)

    # Create figure with 4 subplots (abs, squared on top; early, late on bottom)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))

    # Full timeline absolute difference plot
    for i in range(k_len):
        k_value = k_values[i]
        ax1.plot(x_axis, avg_abs_diff_elo[i], label=f'K={k_value}')

    ax1.set_title('Full Timeline - Absolute Difference')
    ax1.set_xlabel('Number of Games Played')
    ax1.set_ylabel('Average Absolute Difference from True Rating')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Full timeline squared difference plot
    for i in range(k_len):
        k_value = k_values[i]
        ax2.plot(x_axis, avg_sq_diff_elo[i], label=f'K={k_value}')

    ax2.set_title('Full Timeline - Squared Difference')
    ax2.set_xlabel('Number of Games Played')
    ax2.set_ylabel('Average Squared Difference from True Rating')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # First 300 games plot
    for i in range(k_len):
        k_value = k_values[i]
        ax3.plot(x_axis[:301], avg_abs_diff_elo[i, :301], label=f'K={k_value}')

    ax3.set_title('First 300 Games - Absolute Difference')
    ax3.set_xlabel('Number of Games Played')
    ax3.set_ylabel('Average Absolute Difference from True Rating')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Last 300 games plot
    for i in range(k_len):
        k_value = k_values[i]
        ax4.plot(x_axis[-301:], avg_abs_diff_elo[i, -301:], label=f'K={k_value}')

    ax4.set_title('Last 300 Games - Absolute Difference')
    ax4.set_xlabel('Number of Games Played')
    ax4.set_ylabel('Average Absolute Difference from True Rating')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

# Helper function for curve fitting
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Print convergence statistics in a formatted table
def print_convergence_stats(k_values, elo_conv_stats):
    k_len = len(k_values)

    print("Convergence Statistics for Different K Values:")
    print("-" * 80)
    print(f"{'K Value':<10} {'Games to':<15} {'Convergent':<12} {'Min Value':<12} {'Max Value':<12}")
    print(f"{'':10} {'Converge':15} {'Value':12} {'in Zone':12} {'in Zone':12}")
    print("-" * 80)

    for i in range(k_len):
        k_value = k_values[i]
        games, conv_val, low, high = elo_conv_stats[i]
        
        # Format the values, handle None cases
        games_str = f"{games}" if games != -1 else "Didn't conv."  # Shortened and fixed spacing
        conv_str = f"{conv_val:.2f}" if conv_val is not None else "N/A"
        low_str = f"{low:.2f}" if low is not None else "N/A"
        high_str = f"{high:.2f}" if high is not None else "N/A"
        
        print(f"{k_value:<10} {games_str:<15} {conv_str:<12} {low_str:<12} {high_str:<12}")

# Games to converge plot and curve fitting
def graph_games_to_converge(k_values, elo_conv_stats, ax1):
    games_to_converge = np.array([stats[0] if stats[0] != -1 else np.nan for stats in elo_conv_stats])
    valid_indices = ~np.isnan(games_to_converge)
    if np.any(valid_indices):
        x_fit = k_values[valid_indices]
        y_fit = games_to_converge[valid_indices]
        
        # Store R² values for each fit
        fit_r2 = {}
        
        # Try linear and quadratic fits
        for degree in range(1, 3):
            coeffs = np.polyfit(x_fit, y_fit, degree)
            y_pred = np.polyval(coeffs, x_fit)
            r2 = 1 - np.sum((y_fit - y_pred)**2) / np.sum((y_fit - np.mean(y_fit))**2)
            fit_r2[f'{"Linear" if degree==1 else "Quadratic"}'] = r2
        
        # Try exponential fit
        try:
            popt, _ = curve_fit(exp_func, x_fit, y_fit)
            y_pred = exp_func(x_fit, *popt)
            r2 = 1 - np.sum((y_fit - y_pred)**2) / np.sum((y_fit - np.mean(y_fit))**2)
            fit_r2['Exponential'] = r2
        except:
            fit_r2['Exponential'] = float('-inf')
        
        # Print all R² values
        print("\nGames to Converge - R² values:")
        print("-" * 40)
        for fit_type, r2 in fit_r2.items():
            print(f"{fit_type:<15}: {r2:.4f}")
        
        # Plot best fit
        best_fit = max(fit_r2.items(), key=lambda x: x[1])
        ax1.plot(k_values, games_to_converge, 'bo', label='Data')
        x_smooth = np.linspace(min(x_fit), max(x_fit), 100)
        
        if best_fit[0] == 'Linear':
            coeffs = np.polyfit(x_fit, y_fit, 1)
            y_smooth = np.polyval(coeffs, x_smooth)
        elif best_fit[0] == 'Quadratic':
            coeffs = np.polyfit(x_fit, y_fit, 2)
            y_smooth = np.polyval(coeffs, x_smooth)
        else:  # exponential
            popt, _ = curve_fit(exp_func, x_fit, y_fit)
            y_smooth = exp_func(x_smooth, *popt)
        
        ax1.plot(x_smooth, y_smooth, 'b--', label=f'Best fit ({best_fit[0]})')

    ax1.set_title('Games Required to Converge vs K Value')
    ax1.set_xlabel('K Value')
    ax1.set_ylabel('Number of Games to Converge')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

# Convergent values plot and curve fitting
def graph_convergent_values(k_values, elo_conv_stats, ax2):
    convergent_values = np.array([stats[1] if stats[1] is not None else np.nan for stats in elo_conv_stats])
    valid_indices = ~np.isnan(convergent_values)
    if np.any(valid_indices):
        x_fit = k_values[valid_indices]
        y_fit = convergent_values[valid_indices]
        
        # Store R² values for each fit
        fit_r2 = {}
        
        # Try linear and quadratic fits
        for degree in range(1, 3):
            coeffs = np.polyfit(x_fit, y_fit, degree)
            y_pred = np.polyval(coeffs, x_fit)
            r2 = 1 - np.sum((y_fit - y_pred)**2) / np.sum((y_fit - np.mean(y_fit))**2)
            fit_r2[f'{"Linear" if degree==1 else "Quadratic"}'] = r2
        
        # Try exponential fit
        try:
            popt, _ = curve_fit(exp_func, x_fit, y_fit)
            y_pred = exp_func(x_fit, *popt)
            r2 = 1 - np.sum((y_fit - y_pred)**2) / np.sum((y_fit - np.mean(y_fit))**2)
            fit_r2['Exponential'] = r2
        except:
            fit_r2['Exponential'] = float('-inf')
        
        # Print all R² values
        print("\nConvergent Value - R² values:")
        print("-" * 40)
        for fit_type, r2 in fit_r2.items():
            print(f"{fit_type:<15}: {r2:.4f}")
        
        # Plot best fit
        best_fit = max(fit_r2.items(), key=lambda x: x[1])
        ax2.plot(k_values, convergent_values, 'ro', label='Data')
        x_smooth = np.linspace(min(x_fit), max(x_fit), 100)
        
        if best_fit[0] == 'Linear':
            coeffs = np.polyfit(x_fit, y_fit, 1)
            y_smooth = np.polyval(coeffs, x_smooth)
        elif best_fit[0] == 'Quadratic':
            coeffs = np.polyfit(x_fit, y_fit, 2)
            y_smooth = np.polyval(coeffs, x_smooth)
        else:  # exponential
            popt, _ = curve_fit(exp_func, x_fit, y_fit)
            y_smooth = exp_func(x_smooth, *popt)
        
        ax2.plot(x_smooth, y_smooth, 'r--', label=f'Best fit ({best_fit[0]})')

    ax2.set_title('Convergent Rating Difference vs K Value')
    ax2.set_xlabel('K Value')
    ax2.set_ylabel('Average Absolute Rating Difference')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

# Print convergence threshold statistics in a formatted table
def print_convergence_threshold_speed(k_values, elo_avg_conv_25, elo_avg_conv_50, elo_avg_conv_75, elo_avg_conv_100):
    print("Games to Reach Rating Difference Thresholds:")
    print("-" * 70)
    print(f"{'K Value':<10} {'≤ 100':<12} {'≤ 75':<12} {'≤ 50':<12} {'≤ 25':<12}")
    print("-" * 70)

    for i in range(len(k_values)):
        k_value = k_values[i]
        conv_100 = elo_avg_conv_100[i] if elo_avg_conv_100[i] != 0 else "Never"
        conv_75 = elo_avg_conv_75[i] if elo_avg_conv_75[i] != 0 else "Never"
        conv_50 = elo_avg_conv_50[i] if elo_avg_conv_50[i] != 0 else "Never"
        conv_25 = elo_avg_conv_25[i] if elo_avg_conv_25[i] != 0 else "Never"
        
        print(f"{k_value:<10} {str(conv_100):<12} {str(conv_75):<12} {str(conv_50):<12} {str(conv_25):<12}")

# Graph convergence threshold statistics
def graph_convergence_thresholds_speed(k_values, elo_avg_conv_25, elo_avg_conv_50, elo_avg_conv_75, elo_avg_conv_100):
    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot each threshold
    plt.plot(k_values, elo_avg_conv_100, 'bo-', label='≤ 100 rating diff', alpha=0.8)
    plt.plot(k_values, elo_avg_conv_75, 'go-', label='≤ 75 rating diff', alpha=0.8)
    plt.plot(k_values, elo_avg_conv_50, 'ro-', label='≤ 50 rating diff', alpha=0.8)
    plt.plot(k_values, elo_avg_conv_25, 'mo-', label='≤ 25 rating diff', alpha=0.8)

    plt.title('Games Required to Reach Rating Difference Thresholds')
    plt.xlabel('K Value')
    plt.ylabel('Number of Games Played')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add a note about zero values
    plt.text(0.02, 0.98, 'Note: 0 means threshold was never reached', 
            transform=plt.gca().transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# Graph ranking deviation
def graph_ranking_deviation(k_values, elo_ranking_dev):
    # Create plot for ranking deviation
    plt.figure(figsize=(12, 8))

    # Plot ranking deviation
    plt.plot(k_values, elo_ranking_dev, 'ro-', label='Ranking Deviation', linewidth=2, markersize=8)

    plt.title('Ranking Deviation vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Aggregate Ranking Deviation (%)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add note about interpretation
    plt.text(0.02, 0.98, 'Note: Higher ranking deviation indicates greater error in player rankings', 
            transform=plt.gca().transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# Print ranking deviation statistics in a formatted table
def print_ranking_deviation(k_values, elo_ranking_dev):
    print("\nRanking Deviation for Different K Values:")
    print("-" * 50)
    print(f"{'K Value':<10} {'Ranking Deviation (%)':<20}")
    print("-" * 50)
    for i in range(len(k_values)):
        k_value = k_values[i]
        print(f"{k_value:<10} {elo_ranking_dev[i]:<20.2f}")

# Graph rating volatility during adjustment and convergence periods
def graph_rating_volatility(k_values, elo_adj_mse, elo_conv_mse):
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot Adjustment Period MSE
    ax1.plot(k_values, elo_adj_mse, 'bo-', label='Adjustment MSE', linewidth=2, markersize=8)
    ax1.set_title('Rating Adjustment Period Volatility')
    ax1.set_xlabel('K Value')
    ax1.set_ylabel('Mean Squared Error')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot Convergence Period MSE (excluding non-convergent cases)
    valid_indices = np.array(elo_conv_mse) != -1
    ax2.plot(k_values[valid_indices], np.array(elo_conv_mse)[valid_indices], 
            'ro-', label='Convergence MSE', linewidth=2, markersize=8)
    ax2.set_title('Convergence Period Rating Volatility')
    ax2.set_xlabel('K Value')
    ax2.set_ylabel('Mean Squared Error')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # Add notes about interpretation
    ax1.text(0.02, 0.98, 'Note: Higher MSE indicates more volatile rating changes\nduring initial adjustment period', 
            transform=ax1.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.text(0.02, 0.98, 'Note: Higher MSE indicates more volatile rating changes\nafter convergence (excluding non-convergent cases)', 
            transform=ax2.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# Print rating_volatility statistics in formatted tables
def print_rating_volatility(k_values, elo_adj_mse, elo_conv_mse):
    print("\nMSE Values for Different K Values:")
    print("-" * 70)
    print(f"{'K Value':<10} {'Adjustment MSE':<25} {'Convergence MSE':<25}")
    print("-" * 70)

    for i in range(len(k_values)):
        k_value = k_values[i]
        adj_mse = f"{elo_adj_mse[i]:.2f}"
        conv_mse = f"{elo_conv_mse[i]:.2f}" if elo_conv_mse[i] != -1 else "Did not converge"
        print(f"{k_value:<10} {adj_mse:<25} {conv_mse:<25}")
    print("-" * 70)

# Helper function for normalizing statistics
def min_normalize(x):
    # Handle special cases where array contains -1 or nan
    valid_mask = (x != 0) & (x != -1) & (~pd.isna(x))
    if not np.any(valid_mask):
        return np.zeros_like(x)  # Return zeros if no valid data
    
    x_valid = x[valid_mask]
    x_norm = np.zeros_like(x, dtype=float)
    x_norm[valid_mask] = (np.max(x_valid) - x_valid) / (np.max(x_valid) - np.min(x_valid)) if len(x_valid) > 1 else 1.0
    return x_norm

# Print normalized metrics in a formatted table
def print_normed_metrics(k_values, norm_metrics):
    print("\nNormalized Statistics (0-1 scale):")
    print("-" * 180)

    # Print header
    headers = ['K Value'] + list(norm_metrics.keys())
    header_fmt = "{:<8}" + "{:<20}" * len(norm_metrics)
    print(header_fmt.format(*headers))
    print("-" * 180)

    # Print data rows
    row_fmt = "{:<8}" + "{:<20.3f}" * len(norm_metrics)
    for i, k in enumerate(k_values):
        values = [k] + [norm_metrics[metric][i] for metric in norm_metrics.keys()]
        print(row_fmt.format(*values))

    print("-" * 180)

# Print optimal parameter results in a formatted table
def print_optimal_param(sorted_scores):
    print("\nWeighted Combined Scores (higher is better):")
    print("-" * 50)
    print(f"{'K Value':<10} {'Score':<10} {'Rank':<10}")
    print("-" * 50)

    for rank, (k, score) in enumerate(sorted_scores, 1):
        print(f"{k:<10} {score:.3f}    #{rank}")
    print("-" * 50)

# Graph optimal parameter results
def graph_optimal_param(k_values, weighted_scores):
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, weighted_scores, 'bo-', linewidth=2, markersize=8)
    plt.title('Combined Performance Score vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Weighted Score (higher is better)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Annotate the best K value
    best_k_idx = weighted_scores.index(max(weighted_scores))
    plt.annotate(f'Best K = {k_values[best_k_idx]}',
                xy=(k_values[best_k_idx], weighted_scores[best_k_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    plt.show()