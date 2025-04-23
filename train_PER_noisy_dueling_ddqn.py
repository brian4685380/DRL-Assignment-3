from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
from PIL import Image
import torch
from NoisyDuelingQNetwork import NoisyDuelingQNetwork
from constant import *
from utils import *
from PrioritizedBuffer import PrioritizedBuffer
from torch.optim import Adam
import time
from wrappers import wrap_environment


def render_rgb(env):
    return Image.fromarray(env.render(mode='rgb_array'))

env = wrap_environment(
<<<<<<< HEAD
        "SuperMarioBros-v0",
        COMPLEX_MOVEMENT,
        skip=4,
        num_steps=4
=======
    "SuperMarioBros-v0",
      COMPLEX_MOVEMENT,
>>>>>>> 2797b80 (update train_PER_noisy_dueling_ddqn.py)
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
input_shape = env.observation_space.shape
qnet = NoisyDuelingQNetwork(input_shape, env.action_space.n).to(device)
target_qnet = NoisyDuelingQNetwork(input_shape, env.action_space.n).to(device)
qnet_dir = "mario_PER_noisy_dueling_ddqn_model.pth"
target_qnet_dir = "mario_PER_noisy_dueling_ddqn_target_model.pth"

# with open(qnet_dir, 'rb') as f:
#     qnet.load_state_dict(torch.load(f))
# with open(target_qnet_dir, 'rb') as f:
#     target_qnet.load_state_dict(torch.load(f))
replay_buffer = PrioritizedBuffer(capacity=MEMORY_CAPACITY)
optimizer = Adam(qnet.parameters(), lr=LEARNING_RATE)

reward_history = []
best_reward = -np.inf

def update(steps, global_steps):
    if len(replay_buffer) > INITIAL_LEARNING:
        if global_steps % TARGET_UPDATE_FREQUENCY == 0:
            target_qnet.load_state_dict(qnet.state_dict())
        optimizer.zero_grad()
        beta = update_beta(steps)
        states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(BATCH_SIZE, beta)
        
        q_values = qnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_action = qnet(next_states).argmax(dim=1, keepdim=True)
            next_q_values = target_qnet(next_states).gather(1, next_action).squeeze(1)
            target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
            
        td_errors = q_values - target_q_values
        loss = (td_errors.pow(2) * weights).mean()
        loss.backward()
        optimizer.step()
        qnet.reset_noise()
        target_qnet.reset_noise()
        replay_buffer.update_priorities(indices, td_errors)    

if __name__ == "__main__":
    tic = time.time()
    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        global_steps = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = qnet(state_tensor)
            action = q_values.max(dim=1)[1].item()
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1
            global_steps += 1
            update(steps,global_steps)
            steps = 0
        reward_history.append(episode_reward)    
        if episode_reward > best_reward:
            best_reward = episode_reward
            with open(qnet_dir, 'wb') as f:
                torch.save(qnet.state_dict(), f)
            with open(target_qnet_dir, 'wb') as f:
                torch.save(target_qnet.state_dict(), f)
        toc = time.time()
        if episode % 10 == 0:
            average_reward = np.mean(reward_history[-10:])
            print(f"Episode {episode} | average reward: {average_reward:.4f} | reward: {episode_reward:.2f} | time: {toc - tic:.2f}s")
            tic = time.time()
