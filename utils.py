from constant import *
import math
def update_epsilon(episode):
    epsilon_start = EPSILON_START
    epsilon_final = EPSILON_FINAL
    epsilon_decay = EPSILON_DECAY
    return max(epsilon_final, epsilon_start * (epsilon_decay ** episode))

def update_beta(steps):
    beta_start = BETA_START
    beta_frames = BETA_FRAMES
    beta_final = 1.0
    beta = beta_start + (beta_final - beta_start) * steps / beta_frames
    return min(beta, beta_final)