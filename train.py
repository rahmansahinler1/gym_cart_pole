from typing import Tuple, Dict

import gym
import numpy as np
import torch

import pathlib

"""
### Cart Pole Environment ###
* Action space:
    - The agent has four actions available:
        * Push cart to the left = 0
        * Push cart to the right = 1
        
* Observation space:
    - Agent's observation space is a state vector with 4 variables:
        * Cart position (-4.8, 4.8). Terminates (-2.4, 2.4)
        * Cart velocity (-Inf, Inf)
        * Pole angle (-0.418 rad, 0.418 rad) Terminates (-0.2095, 0.2095)
        * Pole angular velocity (-Inf, Inf)
        
* Rewards:
    - Goal is to keep pole upright for as long as possible. 
    - Reward:
        * +1 for every step taken including terminal state.
        * Threshold for reward is 475

* Starting state:
    - All observations assigned uniformly random value in (-0.05, 0.05)
        
* Terminal State:
    - Episode ends if following:
        * Pole angle > +- 0.2095
        * Cart position > +- 2.4 
        * Episode length > 500
"""

class QValue(torch.nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.dense_layer_1 = torch.nn.Linear(state_size, 128)
        self.dense_layer_2 = torch.nn.Linear(128, action_size)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.dense_layer_1(state))
        return self.dense_layer_2(features)

class FixedSizeBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = []
        self.buffer_size = 0
        
    def append(self, item):
        if len(self.items) >= self.max_size:
            self.items[self.buffer_size % self.max_size] = item
        else:
            self.items.append(item)
            
        self.buffer_size += 1
        
    def __str__(self):
        return str(self.items)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]

# Model saving directory
save_dir = pathlib.Path().resolve().joinpath('models')

# Seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Hyperparameters
memory_size = 10000        # Buffer memory size
alpha = 1e-3               # Learning rate
gamma = 0.95               # Dicount factor
n_iteration = 500000       # Total number of iteration
epsilon = 0.1              # Randomness

# Experience Buffer
buffer = FixedSizeBuffer(max_size=memory_size)

# Load the environment
env = gym.make('CartPole-v1')
env.reset(seed=seed)

# Neural Network
value_function = QValue(4, 2)
optimizer = torch.optim.Adam(value_function.parameters(), lr=alpha)

def get_qvalues(state: np.ndarray) -> torch.Tensor:
    """Calculate QValues

    Args:
        state (np.ndarray): state vector

    Returns:
        torch.Tensor: Q values
    """
    state = torch.from_numpy(state).float().unsqueeze(0)
    qvalues = value_function.forward(state)
    return qvalues

def get_action(state: np.ndarray, epsilon: float) -> Tuple[float, float]:
    """Epsilon greedy behaviour policy

    Args:
        state (np.ndarray): current state
        eps (float): current epsilon

    Returns:
        Tuple[float, float]: action, p(action)
    """
    if np.random.rand() < epsilon:
        return np.random.randint(0, 2, size=1).item()
    qvalues = get_qvalues(state)
    return qvalues.max(dim=1).indices.item()

def evaluate() -> float:
    """Evaluate the soft agent for one episode

    Args:
        is_render (bool, optional): render or not. Defaults to False.
        is_save_render (bool, optional): save the render or not. Defaults to False.

    Returns:
        float: episode reward
    """
    eval_env = gym.make('CartPole-v1')
    
    eps_reward = 0
    state, _ = eval_env.reset(seed=seed)
    done = False
    while not done:
        action = get_action(state, epsilon)
        state, reward, done, _, _ = eval_env.step(action)
        eps_reward += reward
    return eps_reward

def sample_from_buffer(batch_size: int) -> Dict[str, torch.Tensor]:
    """Uniformly sample a batch of transitions

    Args:
        batch_size (int): batch size

    Returns:
        Dict[str, torch.Tensor]: Sample of transitions as torch tensor
    """
    indices = np.random.randint(0, len(buffer), size=batch_size)
    items = [buffer[index] for index in indices]
    
    return {
        key: torch.from_numpy(np.array([item[key] for item in items])).float()
        for key in items[0].keys()
    }

def td_loss_fn(sample: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Calculate TD loss for the given batch of transitions

    Args:
        sample (Dict[str, torch.Tensor]): batch of torch transitions

    Returns:
        torch.Tensor: TD error tensor
    """
    next_action = sample["next_action"].long().reshape(-1, 1)
    action = sample["action"].long().reshape(-1, 1)
    reward = sample["reward"].reshape(-1, 1)
    done = sample["done"].reshape(-1, 1)

    next_qvalues = value_function.forward(sample["next_state"])
    selected_next_qvalues = next_qvalues.gather(dim=1, index=next_action)

    target_qvalue = reward + (1 - done) * gamma * selected_next_qvalues  # Bellman equation
    qvalue = value_function.forward(sample["state"]).gather(dim=1, index=action)  # Q(S).gather(.) -> Q(s, a)
    
    td_loss = ((target_qvalue.detach() - qvalue) ** 2)
    
    return td_loss.mean()

if __name__ == '__main__':
    episodic_rewards = []
    episode_reward = 0
    state, _ = env.reset()
    action = get_action(state, epsilon)
    for iter_index in range(1, n_iteration + 1):
        next_state, reward, done, _, _ = env.step(action)
        next_action = get_action(next_state, epsilon)
        episode_reward += reward      
        buffer.append(
            # SARSA
            dict(
                state=state,
                action=action,
                reward=reward,
                done=done,
                next_state=next_state,
                next_action=next_action
            )
        )
        action, state = next_action, next_state
        
        if done:
            # Episode finished
            episodic_rewards.append(episode_reward)
            episode_reward = 0
            state, _ = env.reset()
        
        if not(iter_index % 250000):
            # Save the model for every 250k iteration
            torch.save({
                'iteration': iter_index,
                'model_state_dict': value_function.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': td_loss.item()
            }, save_dir.joinpath(f'model{iter_index}.pt'))
            
        if not(iter_index % 25000):
            # Update user about training
            eval_reward = evaluate()
            avg_train_reward = np.mean(episodic_rewards)
            episodic_rewards = []
            print(f'Iteration: {iter_index}, Evaluation: {eval_reward:.4f}, Training: {avg_train_reward:.4f} Rewards')
        
        if len(buffer) > 1000:
            # Learning happens when enough number of sample gathered
            sample = sample_from_buffer(64)
            optimizer.zero_grad()
            td_loss = td_loss_fn(sample)
            td_loss.backward()
            optimizer.step()
