from typing import Tuple

import torch
import gym
import pathlib
import numpy as np

# Model Path
model_path = pathlib.Path().resolve().joinpath('models/model250000.pt')

class QValue(torch.nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.dense_layer_1 = torch.nn.Linear(state_size, 128)
        self.dense_layer_2 = torch.nn.Linear(128, action_size)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.dense_layer_1(state))
        return self.dense_layer_2(features)
    
def load_model(model_path):
    """Loading the trained model

    Args:
        model_path (pathlib.path): path of the model

    Returns:
        .pt model: model
    """
    model = QValue(4, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()
    return model

def evaluate_model():
    # Same evaluation from training
    eps_reward = 0
    state, _ = env.reset()
    done = False
    while not done:
        action = get_action(state)
        state, reward, done, _, _ = env.step(action)
        eps_reward += reward
    return eps_reward

def get_qvalues(state: np.ndarray) -> torch.Tensor:
    # pre-trained model evaluates the state
    state = torch.from_numpy(state).float().unsqueeze(0)
    qvalues = model.forward(state)
    return qvalues

def get_action(state: np.ndarray) -> Tuple[float, float]:
    # pre-trained model gives the decided action
    qvalues = get_qvalues(state)
    return qvalues.max(dim=1).indices.item()

# Set the environment
env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()
done = False
reward = 0

# Load the model
model = load_model(model_path)

# Evaluation with 10 episodes
for episode_ctr in range(10):
    eval_reward = evaluate_model()
    print(f"Episode: {episode_ctr + 1}, Evaluation Reward: {eval_reward:.4f}")
    