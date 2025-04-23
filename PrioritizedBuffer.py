import numpy as np
import torch

class PrioritizedBuffer:
    def __init__(self, capacity, alpha = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        prios = np.array(self.priorities)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = np.array(next_states)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        indices = torch.tensor(indices, dtype=torch.long).to(self.device)

        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, td_errors, epsilon=1e-5):
        td_errors = td_errors.detach().cpu().numpy()
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + epsilon
            
    def __len__(self):
        return len(self.buffer)
