import numpy as np
import torch

class PrioritizedBuffer:
    def __init__(self, capacity, alpha = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta = 0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        idx = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in idx]

        weights = (len(self.buffer) * probabilities[idx]) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]
        return states, actions, rewards, next_states, dones, idx, weights
    
    def update_priorities(self, idx, priorities):
        for i, p in zip(idx, priorities):
            self.priorities[i] = p
            
    def __len__(self):
        return len(self.buffer)