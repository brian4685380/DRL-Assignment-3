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
        weights = torch.FloatTensor(weights).unsqueeze(1)

        batch = list(zip(*samples))
        states      = torch.cat([
            torch.from_numpy(s.squeeze(0).copy()).unsqueeze(0)
            for s in batch[0]
        ])
        actions     = torch.LongTensor(batch[1]).unsqueeze(1)
        rewards     = torch.FloatTensor(batch[2]).unsqueeze(1)
        next_states = torch.cat([
            torch.from_numpy(s.squeeze(0).copy()).unsqueeze(0)
            for s in batch[3]
        ])
        dones       = torch.FloatTensor(batch[4]).unsqueeze(1)
        return states, actions, rewards, next_states, dones, idx, weights
    
    def update_priorities(self, idx, priorities):
        for i, p in zip(idx, priorities):
            self.priorities[i] = p.item()
            
    def __len__(self):
        return len(self.buffer)