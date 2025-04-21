from collections import deque
import random
import torch
import numpy as np
class ReplayBuffer:
    def __init__(self, capacity):
        # TODO: Initialize the buffer
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: Implement the add method
    # def add(self, ...):
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))


    # TODO: Implement the sample method
    # def sample(self, ...):
    def sample(self, batch_size):
        transition = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transition)
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = np.array(next_states)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)