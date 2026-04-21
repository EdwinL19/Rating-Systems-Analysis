We implement the well-known rating systems Elo, Glicko, and Glicko2 in Python using NumPy vectorization for fast computation. Using these rating systems, we simulate game environments using both skill based and random matchmaking on the scale of 100,000+ players and 1000s of games played for each player.

The most exciting part of this project is our investigation of an unusual divergence of ratings (termed rating drift) under skill based matchmaking over extremely large numbers of games played. 
