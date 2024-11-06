import random
import numpy as np

class GridWorld:
    def __init__(self, width=10, height=10, num_obstacles=10):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        for _ in range(num_obstacles):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            self.grid[y][x] = 1
        self.start = (0, 0)
        self.goal = (width - 1, height - 1)
        
    def is_valid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] != 1
    
    def get_next_state(self, state, action):
        x, y = state
        if action == 'up':
            return (x, max(y-1, 0))
        elif action == 'down':
            return (x, min(y+1, self.height-1))
        elif action == 'left':
            return (max(x-1, 0), y)
        elif action == 'right':
            return (min(x+1, self.width-1), y)


    def get_reward(self, state):
        x, y = state
        if 0 <= x < self.width and 0 <= y < self.height:
            if state == self.goal:
                return 100
            elif self.grid[y][x] == 1:
                return -100
            else:
                return -1
        else:
            return -100
class QLearningAgent:
    def __init__(self,grid_world,alpha=0.1,gamma=0.9,epsilon=0.1):
        self.grid_world=grid_world
        self.q_table=np.zeros((grid_world.height,grid_world.width,4),dtype=np.float32)
        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=epsilon
    def choose_action(self,state):
        if random.uniform(0,1)<self.epsilon:
            return random.choice(['up','down','left','right'])
        else:
            x,y=state
            return['up','down','left','right'][np.argmax(self.q_table[y][x])]
    def update_q_table(self,state,action,reward,next_state):
        x,y=state
        nx,ny=next_state
        self.q_table[y][x][["up","down","left","right"].index(action)]=(1-self.alpha)*self.q_table[y][x][["up","down","left","right"].index(action)]+self.alpha*(reward+self.gamma* max(self.q_table[ny][nx]))
    def train(self,num_episodes):
        for _ in range(num_episodes):
            state=self.grid_world.start
            done=False
            while not done:
                action =self.choose_action(state)
                next_state = self.grid_world.get_next_state(state,action)
                reward=self.grid_world.get_reward(next_state)
                self.update_q_table(state,action,reward,next_state)
                state=next_state
                if state == self.grid_world.goal or self.grid_world.grid[next_state[1]][next_state[0]]==1:
                    done == True
    def get_optimal_policy(self):
        policy = [[None for _ in range(self.grid_world.width)] for _ in range(self.grid_world.height)]
        for y in range(self.grid_world.height):
            for x in range(self.grid_world.width):
                if (x,y) != self.grid_world.goal:
                    policy[y][x] = ['up','down','left','right'][np.argmax(self.q_table[y][x])]
            return policy

grid_world = GridWorld(width=10, height=10, num_obstacles=10)
agent = QLearningAgent(grid_world)
agent.train(100)
optimal_policy = agent.get_optimal_policy()

print("Optimal Policy Matrix:")
for row in optimal_policy:
    print(" ".join([f"{p:5}" for p in row]))