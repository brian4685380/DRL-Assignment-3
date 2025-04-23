import gym
import cv2
import numpy as np
import torch
from NoisyDuelingQNetwork import NoisyDuelingQNetwork

qnet_dir = "mario_PER_noisy_dueling_ddqn_model.pth"
input_shape = (1, 84, 84)
action_space = 12
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent(object):
    """DQN Agent for Super Mario Bros."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(action_space)
        self.device = device
        self.qnet = NoisyDuelingQNetwork(input_shape, action_space).to(self.device)
        self.qnet.load_state_dict(torch.load(qnet_dir, map_location=self.device))
        self.qnet.eval()

    def act(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)  # RGB → gray
        observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        observation = observation[:, :, None]  # HWC → HWC1
        observation = np.moveaxis(observation, 2, 0)  # HWC → CHW
        observation = np.array(observation).astype(np.float32) / 255.0

        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.qnet(state_tensor)
        action = q_values.argmax(dim=1).item()
        return action