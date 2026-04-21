"""
Contextual Bandit Implementation (LinUCB)
==========================================
Used by the Curriculum Agent to select optimal topics for each learner.

LinUCB uses linear payoff models with upper confidence bounds to balance
exploration of new topics with exploitation of known effective ones.

Reference: Li et al., "A Contextual-Bandit Approach to Personalized News
Article Recommendation", WWW 2010
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class LinUCBConfig:
    """Configuration for LinUCB algorithm."""
    context_dim: int = 16
    n_arms: int = 10
    alpha: float = 1.5          # Exploration parameter
    alpha_decay: float = 0.999  # Decay rate for exploration
    min_alpha: float = 0.1      # Minimum exploration
    regularization: float = 1.0  # Ridge regression lambda
    seed: int = 42


class LinUCBAgent:
    """
    LinUCB Contextual Bandit for curriculum topic selection.
    
    Context: Learner state (knowledge, engagement, learning style, history)
    Arms: Available topics/modules to teach
    Reward: Learning gain from the selected topic
    
    Uses disjoint linear models - one per arm - with UCB exploration.
    """

    def __init__(self, config: LinUCBConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.alpha = config.alpha
        d = config.context_dim

        # Per-arm parameters
        self.A = {}  # d x d matrices
        self.b = {}  # d x 1 vectors
        self.theta = {}  # Learned parameters

        for arm in range(config.n_arms):
            self.A[arm] = config.regularization * np.eye(d)
            self.b[arm] = np.zeros((d, 1))
            self.theta[arm] = np.zeros((d, 1))

        # Statistics
        self.arm_counts = np.zeros(config.n_arms)
        self.arm_rewards = np.zeros(config.n_arms)
        self.total_reward = 0.0
        self.total_steps = 0
        self.history: List[Dict] = []
        self.training_stats: List[Dict] = []

    def select_arm(self, context: np.ndarray, available_arms: Optional[List[int]] = None) -> Tuple[int, Dict]:
        """
        Select the best arm given context using UCB.
        
        Returns:
            arm: Selected arm index
            info: Dictionary with UCB scores and details
        """
        context = context.reshape(-1, 1)
        d = self.config.context_dim

        if available_arms is None:
            available_arms = list(range(self.config.n_arms))

        ucb_scores = {}
        predicted_rewards = {}
        uncertainties = {}

        for arm in available_arms:
            A_inv = np.linalg.solve(self.A[arm], np.eye(d))
            self.theta[arm] = A_inv @ self.b[arm]

            # Predicted reward
            pred = float((self.theta[arm].T @ context).item())
            predicted_rewards[arm] = pred

            # Uncertainty (confidence width)
            uncertainty = self.alpha * np.sqrt(float((context.T @ A_inv @ context).item()))
            uncertainties[arm] = uncertainty

            # UCB score
            ucb_scores[arm] = pred + uncertainty

        # Select arm with highest UCB
        best_arm = max(available_arms, key=lambda a: ucb_scores[a])

        info = {
            'ucb_scores': ucb_scores,
            'predicted_rewards': predicted_rewards,
            'uncertainties': uncertainties,
            'alpha': self.alpha,
            'selected_arm': best_arm,
        }

        return best_arm, info

    def update(self, arm: int, context: np.ndarray, reward: float):
        """Update model for the selected arm given observed reward."""
        context = context.reshape(-1, 1)

        # Sherman-Morrison update for efficiency
        self.A[arm] += context @ context.T
        self.b[arm] += reward * context

        # Update statistics
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        self.total_reward += reward
        self.total_steps += 1

        # Decay exploration
        self.alpha = max(self.config.min_alpha,
                         self.alpha * self.config.alpha_decay)

        # Record history
        self.history.append({
            'step': self.total_steps,
            'arm': arm,
            'reward': reward,
            'alpha': self.alpha,
            'cumulative_reward': self.total_reward,
        })

    def get_arm_statistics(self) -> Dict[int, Dict]:
        """Get statistics for each arm."""
        stats = {}
        for arm in range(self.config.n_arms):
            count = self.arm_counts[arm]
            stats[arm] = {
                'count': int(count),
                'mean_reward': self.arm_rewards[arm] / max(count, 1),
                'total_reward': self.arm_rewards[arm],
                'theta_norm': float(np.linalg.norm(self.theta[arm])),
            }
        return stats

    def compute_regret(self, optimal_rewards: List[float]) -> np.ndarray:
        """Compute cumulative regret given known optimal rewards."""
        actual_rewards = [h['reward'] for h in self.history]
        regret = np.cumsum(optimal_rewards[:len(actual_rewards)]) - np.cumsum(actual_rewards)
        return regret

    def get_training_summary(self) -> Dict:
        """Get summary of training progress."""
        if not self.history:
            return {}

        recent = self.history[-100:] if len(self.history) >= 100 else self.history
        return {
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'mean_reward': self.total_reward / max(self.total_steps, 1),
            'recent_mean_reward': np.mean([h['reward'] for h in recent]),
            'current_alpha': self.alpha,
            'arm_distribution': {int(k): int(v) for k, v in enumerate(self.arm_counts)},
        }


class HybridBandit(LinUCBAgent):
    """
    Hybrid LinUCB with shared and per-arm features.
    
    Extends LinUCB to incorporate both shared context features
    (learner profile) and arm-specific features (topic properties).
    """

    def __init__(self, config: LinUCBConfig, arm_feature_dim: int = 8):
        super().__init__(config)
        self.arm_feature_dim = arm_feature_dim
        d = config.context_dim
        k = arm_feature_dim

        # Shared parameter
        self.A_shared = config.regularization * np.eye(k)
        self.b_shared = np.zeros((k, 1))
        self.beta = np.zeros((k, 1))

        # Per-arm interaction matrices
        self.B = {}
        for arm in range(config.n_arms):
            self.B[arm] = np.zeros((d, k))

    def select_arm_hybrid(self, context: np.ndarray,
                          arm_features: Dict[int, np.ndarray],
                          available_arms: Optional[List[int]] = None) -> Tuple[int, Dict]:
        """Select arm using hybrid model with arm features."""
        context = context.reshape(-1, 1)
        d = self.config.context_dim

        if available_arms is None:
            available_arms = list(range(self.config.n_arms))

        ucb_scores = {}
        for arm in available_arms:
            if arm not in arm_features:
                continue

            z = arm_features[arm].reshape(-1, 1)
            A_inv = np.linalg.solve(self.A[arm], np.eye(d))
            A_shared_inv = np.linalg.solve(self.A_shared, np.eye(self.arm_feature_dim))

            self.theta[arm] = A_inv @ (self.b[arm] - self.B[arm] @ self.beta)
            self.beta = A_shared_inv @ self.b_shared

            # Predicted reward with both components
            pred = float((z.T @ self.beta + context.T @ self.theta[arm]).item())

            # Combined uncertainty
            s_term = float((z.T @ A_shared_inv @ z).item())
            x_term = float((context.T @ A_inv @ context).item())
            cross_term = float((2 * z.T @ A_shared_inv @ self.B[arm].T @ A_inv @ context).item())

            uncertainty = self.alpha * np.sqrt(max(s_term + x_term - cross_term, 1e-10))
            ucb_scores[arm] = pred + uncertainty

        best_arm = max(available_arms, key=lambda a: ucb_scores.get(a, -np.inf))
        return best_arm, {'ucb_scores': ucb_scores}

    def update_hybrid(self, arm: int, context: np.ndarray,
                      arm_feature: np.ndarray, reward: float):
        """Update hybrid model."""
        context = context.reshape(-1, 1)
        z = arm_feature.reshape(-1, 1)
        d = self.config.context_dim

        A_inv = np.linalg.solve(self.A[arm], np.eye(d))

        self.A_shared += self.B[arm].T @ A_inv @ self.B[arm]
        self.b_shared += self.B[arm].T @ A_inv @ self.b[arm]

        self.A[arm] += context @ context.T
        self.B[arm] += context @ z.T
        self.b[arm] += reward * context

        A_inv_new = np.linalg.solve(self.A[arm], np.eye(d))
        self.A_shared += z @ z.T - self.B[arm].T @ A_inv_new @ self.B[arm]
        self.b_shared += reward * z - self.B[arm].T @ A_inv_new @ self.b[arm]

        # Update stats
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        self.total_reward += reward
        self.total_steps += 1
        self.alpha = max(self.config.min_alpha, self.alpha * self.config.alpha_decay)
