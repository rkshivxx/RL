import gym
import numpy as np
from collections import deque
import random

class MovieEnv(gym.Env):
    def __init__(self, movie_data, num_users, num_movies):
        self.movie_data = movie_data
        self.num_users = num_users
        self.num_movies = num_movies
        self.action_space = gym.spaces.Discrete(self.num_movies)
        self.observation_space = gym.spaces.Discrete(self.num_users)
        self.user_history = [deque(maxlen=10) for _ in range(self.num_users)]

    def step(self, action):
        user = self.state
        movie = action
        reward = self.movie_data[user][movie]
        self.user_history[user].append(movie)
        self.state = np.random.randint(self.num_users)
        return self.state, reward, False, {}

    def reset(self):
        self.state = np.random.randint(self.num_users)
        return self.state

class DQNAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, lr=0.001):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.replay_buffer = deque(maxlen=10000)

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.env.observation_space.n, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(optimizer=Adam(lr=self.lr), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(np.array([state]))[0])

    def replay(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        minibatch = random.sample(self.replay_buffer, batch_size)
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])

        target_q_values = self.target_model.predict(next_states)
        target_values = rewards + self.gamma * np.max(target_q_values, axis=1)

        q_values = self.model.predict(states)
        q_values[np.arange(batch_size), actions] = target_values

        self.model.fit(states, q_values, epochs=1, verbose=0)

    def train(self, num_episodes, batch_size=32):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.append((state, action, reward, next_state))
                state = next_state
                self.replay(batch_size)

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.update_target_model()

movie_data = np.random.rand(100, 1000)
env = MovieEnv(movie_data, 100, 1000)
agent = DQNAgent(env)
agent.train(num_episodes=100, batch_size=32)
