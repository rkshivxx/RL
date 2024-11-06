# 1.Grid world reinforcement learning
This will create a `GridWorld` instance, train the `QLearningAgent`, and print the optimal policy matrix.

## Customization

You can customize the grid world by modifying the parameters in the `GridWorld` class constructor:

- `width`: The width of the grid (default is 10).
- `height`: The height of the grid (default is 10).
- `num_obstacles`: The number of obstacles to be placed in the grid (default is 10).

You can also experiment with the hyperparameters of the `QLearningAgent` class:

- `alpha`: The learning rate (default is 0.1).
- `gamma`: The discount factor (default is 0.9).
- `epsilon`: The exploration rate (default is 0.1).

# 2.Movie Recommendation DQN

## Environment Details
The MovieEnv class simulates a `movie recommendation environment` with users and movies. Key features include:

- `Actions`: Choose from available movies to recommend.
-`Observations`: Represent users in the system.
-`Rewards`: Based on user preferences from the movie_data matrix.
## Class Parameters
-`movie_data`: A 2D numpy array with user interactions with movies.
-`num_users`: Total number of users.
-`num_movies`: Total number of movies.
## Methods
-`step(action)`: Takes an action (movie recommendation), returns the next user, reward, and done flag.
-`reset`: Resets the environment and selects a new random user.
## Agent Details
The DQNAgent uses a deep Q-network to optimize recommendations.

# Class Parameters
-`gamma`: Discount factor for future rewards.
-`epsilon`: Exploration rate for epsilon-greedy strategy.
-`epsilon_decay`: Rate at which epsilon decreases.
-`epsilon_min`: Minimum value of epsilon.
-`lr`: Learning rate for the neural network.
## Methods
-`act`: Chooses an action based on current policy.
-`replay`: Trains the model using experience replay.
-`train`: Trains the agent for multiple episodes.
## Customization
You can customize the movie recommendation environment by modifying parameters in the MovieEnv class constructor:

-`num_users`: Number of users in the environment (default: 100).
-`num_movies`: Number of movies available (default: 1000).
Additionally, experiment with the hyperparameters of the DQNAgent:

-`gamma`: Discount factor (default: 0.99).
-`epsilon`: Exploration rate (default: 1.0, decaying with episodes).
-`epsilon_decay`: Epsilon decay rate (default: 0.995).
-`epsilon_min`: Minimum value for epsilon (default: 0.01).
-`lr`: Learning rate (default: 0.001).
