from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
from PIL import Image
import torch
from QNetwork import QNetwork
from constant import *
from utils import *
# from PrioritizedBuffer import PrioritizedBuffer
from ReplayBuffer import ReplayBuffer
from torch.optim import Adam
import time
from wrappers import wrap_environment


def render_rgb(env):
    return Image.fromarray(env.render(mode='rgb_array'))

env = wrap_environment(
    "SuperMarioBros-v0",
      COMPLEX_MOVEMENT,
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
input_shape = env.observation_space.shape
qnet = QNetwork(input_shape, env.action_space.n).to(device)
target_qnet = QNetwork(input_shape, env.action_space.n).to(device)
qnet_dir = "mario_dqn_model.pth"
target_qnet_dir = "mario_dqn_target_model.pth"

# replay_buffer = PrioritizedBuffer(capacity=MEMORY_CAPACITY)
replay_buffer = ReplayBuffer(capacity=MEMORY_CAPACITY)
optimizer = Adam(qnet.parameters(), lr=LEARNING_RATE)

reward_history = []
best_average_reward = -np.inf

def update(steps):
    if len(replay_buffer) > INITIAL_LEARNING:
        if steps % TARGET_UPDATE_FREQUENCY == 0:
            target_qnet.load_state_dict(qnet.state_dict())
        optimizer.zero_grad()
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
        
        q_values = qnet(states).gather(1, actions.unsqueeze(1)).squeeze(1).to(device)
        with torch.no_grad():
            next_q_values = target_qnet(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        loss = (q_values - target_q_values).pow(2).mean()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    tic = time.time()
    state = env.reset()
    for episode in range(NUM_EPISODES):
        print(f"Episode {episode + 1}/{NUM_EPISODES}")
        env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        epsilon = update_epsilon(episode)
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = qnet(state_tensor)
                action = q_values.max(dim=1)[1].item()
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1
            update(steps)
        print(f"Episode {episode + 1} | Epsilon: {epsilon:.4f} | Steps: {steps} | Reward: {episode_reward:.2f}")
        reward_history.append(episode_reward)    
        toc = time.time()
        average_reward = np.mean(reward_history[-100:])
        if episode % 2 == 0:
            if average_reward > best_average_reward:
                best_average_reward = average_reward
                torch.save(qnet.state_dict(), qnet_dir)
                torch.save(target_qnet.state_dict(), target_qnet_dir)
        print(f"Episode {episode} | average reward: {average_reward:.4f} | reward: {episode_reward:.2f} | time: {toc - tic:.2f}s")
        tic = time.time()
        pass