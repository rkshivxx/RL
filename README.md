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

## Dependencies

The script uses the following Python libraries:

- `random`
- `numpy`
