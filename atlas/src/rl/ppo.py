"""
Proximal Policy Optimization (PPO) Implementation
===================================================
Used by the Pedagogy Agent to learn optimal teaching strategies.

The PPO algorithm clips the policy ratio to prevent destructively large
policy updates, providing stable training for the teaching strategy.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""
    state_dim: int = 32
    action_dim: int = 8
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 32
    seed: int = 42


class NeuralNetwork:
    """Simple feedforward neural network with NumPy."""

    def __init__(self, layer_sizes: List[int], seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # He initialization
            w = self.rng.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass returning output and intermediate activations."""
        activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)  # ReLU
            activations.append(x)
        return x, activations

    def get_params(self) -> List[np.ndarray]:
        return [p.copy() for p in self.weights + self.biases]

    def set_params(self, params: List[np.ndarray]):
        n = len(self.weights)
        self.weights = [p.copy() for p in params[:n]]
        self.biases = [p.copy() for p in params[n:]]


class PPOAgent:
    """
    PPO Agent for learning teaching strategies.
    
    State: Learner profile (knowledge levels, engagement, history)
    Action: Teaching strategy (explanation type, difficulty, pacing, examples)
    Reward: Learning gain + engagement score
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        # Policy network: state -> action logits
        self.policy_net = NeuralNetwork(
            [config.state_dim, config.hidden_dim, config.hidden_dim, config.action_dim],
            seed=config.seed
        )

        # Value network: state -> value estimate
        self.value_net = NeuralNetwork(
            [config.state_dim, config.hidden_dim, config.hidden_dim, 1],
            seed=config.seed + 1
        )

        # Trajectory buffer
        self.buffer = TrajectoryBuffer()

        # Training statistics
        self.training_stats: List[Dict] = []
        self.total_steps = 0

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """Select action using current policy."""
        state = np.atleast_2d(state)
        logits, _ = self.policy_net.forward(state)
        probs = self.softmax(logits[0])

        if deterministic:
            action = np.argmax(probs)
        else:
            action = self.rng.choice(len(probs), p=probs)

        log_prob = np.log(probs[action] + 1e-8)
        return int(action), float(log_prob)

    def get_value(self, state: np.ndarray) -> float:
        """Estimate state value."""
        state = np.atleast_2d(state)
        value, _ = self.value_net.forward(state)
        return float(value[0, 0])

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """Store a transition in the buffer."""
        self.buffer.store(state, action, reward, next_state, done, log_prob)
        self.total_steps += 1

    def compute_gae(self, rewards, values, next_values, dones) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        gamma = self.config.gamma
        lam = self.config.gae_lambda
        T = len(rewards)

        advantages = np.zeros(T)
        last_gae = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self) -> Dict[str, float]:
        """Perform PPO update on collected trajectories."""
        if len(self.buffer) < self.config.batch_size:
            return {}

        data = self.buffer.get_all()
        states = np.array(data['states'])
        actions = np.array(data['actions'])
        rewards = np.array(data['rewards'])
        next_states = np.array(data['next_states'])
        dones = np.array(data['dones'])
        old_log_probs = np.array(data['log_probs'])

        # Compute values
        values = np.array([self.get_value(s) for s in states])
        next_values = np.array([self.get_value(s) for s in next_states])

        # GAE
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.config.n_epochs):
            indices = self.rng.permutation(len(states))

            for start in range(0, len(states), self.config.batch_size):
                end = min(start + self.config.batch_size, len(states))
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]

                # Current policy
                policy_loss, value_loss, entropy = self._compute_losses(
                    batch_states, batch_actions, batch_advantages,
                    batch_returns, batch_old_log_probs
                )

                # Gradient update via finite differences (NumPy-based)
                self._update_networks(
                    batch_states, batch_actions, batch_advantages,
                    batch_returns, batch_old_log_probs
                )

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                n_updates += 1

        self.buffer.clear()

        stats = {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'buffer_size': len(data['states']),
            'mean_reward': float(np.mean(rewards)),
            'mean_advantage': float(np.mean(advantages)),
        }
        self.training_stats.append(stats)
        return stats

    def _compute_losses(self, states, actions, advantages, returns, old_log_probs):
        """Compute PPO losses."""
        # Policy loss
        policy_loss = 0
        entropy = 0
        for i in range(len(states)):
            logits, _ = self.policy_net.forward(states[i:i+1])
            probs = self.softmax(logits[0])
            new_log_prob = np.log(probs[actions[i]] + 1e-8)

            ratio = np.exp(new_log_prob - old_log_probs[i])
            clipped_ratio = np.clip(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)

            policy_loss -= min(ratio * advantages[i], clipped_ratio * advantages[i])
            entropy -= np.sum(probs * np.log(probs + 1e-8))

        policy_loss /= len(states)
        entropy /= len(states)

        # Value loss
        value_loss = 0
        for i in range(len(states)):
            v, _ = self.value_net.forward(states[i:i+1])
            value_loss += (v[0, 0] - returns[i]) ** 2
        value_loss /= (2 * len(states))

        return float(policy_loss), float(value_loss), float(entropy)

    def _update_networks(self, states, actions, advantages, returns, old_log_probs):
        """
        Update networks using advantage-weighted direct logit updates.
        
        For the policy: we directly adjust the final layer weights to increase
        probability of actions with positive advantage and decrease for negative.
        This is equivalent to a simplified policy gradient update.
        
        For the value: we use direct regression gradient on the final layer.
        """
        lr = self.config.learning_rate
        clip_eps = self.config.clip_epsilon
        
        # ---- POLICY UPDATE ----
        # Compute per-sample gradients on final layer only (efficient)
        final_w = self.policy_net.weights[-1]  # (hidden, action_dim)
        final_b = self.policy_net.biases[-1]   # (action_dim,)
        
        w_grad = np.zeros_like(final_w)
        b_grad = np.zeros_like(final_b)
        
        for i in range(len(states)):
            logits, activations = self.policy_net.forward(states[i:i+1])
            probs = self.softmax(logits[0])
            
            new_log_prob = np.log(probs[actions[i]] + 1e-8)
            ratio = np.exp(new_log_prob - old_log_probs[i])
            
            # PPO clipping check
            if advantages[i] > 0:
                if ratio > 1 + clip_eps:
                    continue  # Clipped, skip update
            else:
                if ratio < 1 - clip_eps:
                    continue  # Clipped, skip update
            
            # Compute softmax gradient: d(log pi(a|s))/d(logit_a) = 1 - pi(a)
            # d(log pi(a|s))/d(logit_j) = -pi(j) for j != a
            grad_logits = -probs.copy()
            grad_logits[actions[i]] += 1.0  # = 1 - pi(a) for selected action
            
            # Scale by advantage
            grad_logits *= advantages[i]
            
            # Add entropy bonus gradient
            entropy_grad = -(1 + np.log(probs + 1e-8))
            grad_logits += self.config.entropy_coeff * entropy_grad
            
            # Hidden activations (input to final layer)
            hidden = activations[-2]  # Shape: (1, hidden_dim)
            
            # Accumulate gradients
            w_grad += hidden.T @ grad_logits.reshape(1, -1)
            b_grad += grad_logits
        
        # Apply update (gradient ascent for policy)
        n = max(len(states), 1)
        self.policy_net.weights[-1] += lr * w_grad / n
        self.policy_net.biases[-1] += lr * b_grad / n
        
        # Also update second-to-last layer with smaller step
        if len(self.policy_net.weights) >= 2:
            second_w = self.policy_net.weights[-2]
            second_b = self.policy_net.biases[-2]
            sw_grad = np.zeros_like(second_w)
            sb_grad = np.zeros_like(second_b)
            
            for i in range(len(states)):
                logits, activations = self.policy_net.forward(states[i:i+1])
                probs = self.softmax(logits[0])
                
                grad_logits = -probs.copy()
                grad_logits[actions[i]] += 1.0
                grad_logits *= advantages[i]
                
                # Backprop through final layer
                hidden_grad = (grad_logits.reshape(1, -1) @ self.policy_net.weights[-1].T)
                # ReLU gradient
                hidden_grad *= (activations[-2] > 0).astype(float)
                
                prev_act = activations[-3]  # Input to second-to-last
                sw_grad += prev_act.T @ hidden_grad
                sb_grad += hidden_grad[0]
            
            self.policy_net.weights[-2] += lr * 0.3 * sw_grad / n
            self.policy_net.biases[-2] += lr * 0.3 * sb_grad / n
        
        # ---- VALUE UPDATE ----
        v_final_w = self.value_net.weights[-1]
        v_final_b = self.value_net.biases[-1]
        
        vw_grad = np.zeros_like(v_final_w)
        vb_grad = np.zeros_like(v_final_b)
        
        for i in range(len(states)):
            v_out, v_acts = self.value_net.forward(states[i:i+1])
            td_error = returns[i] - v_out[0, 0]
            
            hidden = v_acts[-2]
            vw_grad += td_error * hidden.T
            vb_grad += td_error * np.ones(1)
        
        self.value_net.weights[-1] += lr * self.config.value_coeff * vw_grad / n
        self.value_net.biases[-1] += lr * self.config.value_coeff * vb_grad.reshape(-1) / n


class TrajectoryBuffer:
    """Buffer for storing trajectories."""

    def __init__(self):
        self.clear()

    def store(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state.copy() if isinstance(state, np.ndarray) else np.array(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state.copy() if isinstance(next_state, np.ndarray) else np.array(next_state))
        self.dones.append(float(done))
        self.log_probs.append(log_prob)

    def get_all(self) -> Dict:
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_states': self.next_states,
            'dones': self.dones,
            'log_probs': self.log_probs,
        }

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def __len__(self):
        return len(self.states)
