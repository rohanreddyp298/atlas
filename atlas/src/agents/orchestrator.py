"""
ATLAS Multi-Agent Orchestrator
===============================
Coordinates the Curriculum Agent (LinUCB), Pedagogy Agent (PPO),
Assessment Agent, and Difficulty Agent to deliver optimal tutoring.

Architecture:
    Orchestrator
    ├── CurriculumAgent (LinUCB) → selects WHAT to teach
    ├── PedagogyAgent (PPO) → selects HOW to teach
    ├── DifficultyAgent (rule-based + adaptive) → calibrates challenge level
    └── AssessmentAgent (Bayesian) → evaluates understanding
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from ..rl.ppo import PPOAgent, PPOConfig
from ..rl.contextual_bandit import LinUCBAgent, LinUCBConfig, HybridBandit
from ..environment.tutorial_env import (
    TutorialEnvironment, LearnerProfile, TeachingStrategy, LearnerType
)


@dataclass
class OrchestratorConfig:
    """Configuration for the ATLAS orchestrator."""
    n_topics: int = 10
    ppo_config: PPOConfig = None
    bandit_config: LinUCBConfig = None
    use_hybrid_bandit: bool = True
    assessment_window: int = 5
    difficulty_adaptation_rate: float = 0.1
    seed: int = 42

    def __post_init__(self):
        if self.ppo_config is None:
            self.ppo_config = PPOConfig(
                state_dim=32,
                action_dim=len(TeachingStrategy),
                hidden_dim=64,
                learning_rate=3e-4,
                seed=self.seed,
            )
        if self.bandit_config is None:
            self.bandit_config = LinUCBConfig(
                context_dim=16,
                n_arms=self.n_topics,
                alpha=1.5,
                seed=self.seed + 100,
            )


class AssessmentAgent:
    """
    Bayesian assessment of learner understanding.
    
    Maintains a belief distribution over each learner's true knowledge
    and updates it based on observed performance.
    """

    def __init__(self, n_topics: int, window: int = 5):
        self.n_topics = n_topics
        self.window = window
        self.beliefs = {}  # learner_id -> {topic: (alpha, beta)}

    def initialize_learner(self, learner_id: str):
        """Initialize uniform Beta priors."""
        self.beliefs[learner_id] = {
            t: {'alpha': 1.0, 'beta': 1.0} for t in range(self.n_topics)
        }

    def update_belief(self, learner_id: str, topic: int, success: float):
        """Update Beta posterior with observation."""
        if learner_id not in self.beliefs:
            self.initialize_learner(learner_id)

        b = self.beliefs[learner_id][topic]
        b['alpha'] += success
        b['beta'] += (1 - success)

    def get_estimated_knowledge(self, learner_id: str) -> np.ndarray:
        """Get MAP estimate of knowledge per topic."""
        if learner_id not in self.beliefs:
            return np.zeros(self.n_topics)

        estimates = np.zeros(self.n_topics)
        for t in range(self.n_topics):
            b = self.beliefs[learner_id][t]
            # MAP of Beta distribution
            estimates[t] = (b['alpha'] - 1) / max(b['alpha'] + b['beta'] - 2, 1)
        return np.clip(estimates, 0, 1)

    def get_uncertainty(self, learner_id: str) -> np.ndarray:
        """Get uncertainty (variance) of knowledge estimate."""
        if learner_id not in self.beliefs:
            return np.ones(self.n_topics) * 0.25

        uncertainty = np.zeros(self.n_topics)
        for t in range(self.n_topics):
            b = self.beliefs[learner_id][t]
            a, bt = b['alpha'], b['beta']
            uncertainty[t] = (a * bt) / ((a + bt) ** 2 * (a + bt + 1))
        return uncertainty


class DifficultyAgent:
    """
    Adaptive difficulty calibration.
    
    Uses a target success rate (typically 0.7-0.8 for optimal learning)
    and adjusts difficulty recommendations based on recent performance.
    """

    def __init__(self, target_success: float = 0.75, adaptation_rate: float = 0.1):
        self.target_success = target_success
        self.adaptation_rate = adaptation_rate
        self.difficulty_offsets = {}

    def get_difficulty_adjustment(self, learner_id: str,
                                  recent_performance: List[float]) -> float:
        """Compute difficulty adjustment based on performance."""
        if learner_id not in self.difficulty_offsets:
            self.difficulty_offsets[learner_id] = 0.0

        if len(recent_performance) < 2:
            return self.difficulty_offsets[learner_id]

        mean_performance = np.mean(recent_performance[-5:])
        error = mean_performance - self.target_success

        self.difficulty_offsets[learner_id] += self.adaptation_rate * error
        self.difficulty_offsets[learner_id] = np.clip(
            self.difficulty_offsets[learner_id], -0.3, 0.3
        )

        return self.difficulty_offsets[learner_id]

    def adjust_strategy(self, base_strategy: TeachingStrategy,
                        difficulty_offset: float) -> TeachingStrategy:
        """Modify strategy based on difficulty adjustment."""
        if difficulty_offset > 0.15:
            # Learner is doing well, increase challenge
            if base_strategy == TeachingStrategy.PRACTICE_EASY:
                return TeachingStrategy.PRACTICE_MEDIUM
            elif base_strategy == TeachingStrategy.PRACTICE_MEDIUM:
                return TeachingStrategy.PRACTICE_HARD
        elif difficulty_offset < -0.15:
            # Learner is struggling, decrease challenge
            if base_strategy == TeachingStrategy.PRACTICE_HARD:
                return TeachingStrategy.PRACTICE_MEDIUM
            elif base_strategy == TeachingStrategy.PRACTICE_MEDIUM:
                return TeachingStrategy.PRACTICE_EASY
            elif base_strategy == TeachingStrategy.LECTURE:
                return TeachingStrategy.WORKED_EXAMPLE

        return base_strategy


class ATLASOrchestrator:
    """
    Main orchestrator coordinating all agents.
    
    Pipeline per step:
    1. AssessmentAgent evaluates learner state
    2. CurriculumAgent (LinUCB) selects topic
    3. PedagogyAgent (PPO) selects teaching strategy
    4. DifficultyAgent adjusts strategy difficulty
    5. Environment executes the interaction
    6. All agents receive feedback and learn
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        # Environment
        self.env = TutorialEnvironment(
            n_topics=config.n_topics, seed=config.seed
        )

        # Agents
        self.curriculum_agent = HybridBandit(
            config.bandit_config,
            arm_feature_dim=8
        ) if config.use_hybrid_bandit else LinUCBAgent(config.bandit_config)

        self.pedagogy_agent = PPOAgent(config.ppo_config)
        self.assessment_agent = AssessmentAgent(config.n_topics, config.assessment_window)
        self.difficulty_agent = DifficultyAgent()

        # Logging
        self.episode_logs: List[Dict] = []
        self.step_logs: List[Dict] = []

    def run_episode(self, learner: Optional[LearnerProfile] = None,
                    max_steps: int = 50) -> Dict:
        """Run a complete tutoring episode."""
        if learner is None:
            learner = self.env.create_learner()

        self.assessment_agent.initialize_learner(learner.learner_id)

        episode_reward = 0.0
        episode_gains = []
        knowledge_trajectory = [learner.knowledge.copy()]

        for step in range(max_steps):
            # 1. Get states
            learner_state = self.env.get_learner_state(learner)
            bandit_context = self.env.get_context_for_bandit(learner)

            # 2. Determine available topics (prerequisite check)
            available_topics = self._get_available_topics(learner)
            if not available_topics:
                available_topics = list(range(min(3, self.config.n_topics)))

            # 3. Curriculum Agent selects topic
            if isinstance(self.curriculum_agent, HybridBandit):
                arm_features = {
                    t: self.env.topics[t].feature_vector
                    for t in available_topics
                }
                topic_id, bandit_info = self.curriculum_agent.select_arm_hybrid(
                    bandit_context, arm_features, available_topics
                )
            else:
                topic_id, bandit_info = self.curriculum_agent.select_arm(
                    bandit_context, available_topics
                )

            # 4. Pedagogy Agent selects strategy
            action, log_prob = self.pedagogy_agent.get_action(learner_state)
            strategy = TeachingStrategy(action)

            # 5. Difficulty adjustment
            recent_gains = [h.get('learning_gain', 0.5) for h in learner.session_history[-5:]]
            diff_offset = self.difficulty_agent.get_difficulty_adjustment(
                learner.learner_id, recent_gains
            )
            strategy = self.difficulty_agent.adjust_strategy(strategy, diff_offset)

            # 6. Execute in environment
            reward, info = self.env.step(learner, topic_id, strategy)

            # 7. Get next state
            next_state = self.env.get_learner_state(learner)
            done = self.env.is_mastered(learner) or step == max_steps - 1

            # 8. Update all agents
            # PPO
            self.pedagogy_agent.store_transition(
                learner_state, int(strategy), reward, next_state, done, log_prob
            )

            # Bandit
            bandit_reward = info['learning_gain'] * 10  # Scale for bandit
            if isinstance(self.curriculum_agent, HybridBandit):
                self.curriculum_agent.update_hybrid(
                    topic_id, bandit_context,
                    self.env.topics[topic_id].feature_vector, bandit_reward
                )
            else:
                self.curriculum_agent.update(topic_id, bandit_context, bandit_reward)

            # Assessment
            success = min(1.0, info['learning_gain'] / 0.1)
            self.assessment_agent.update_belief(learner.learner_id, topic_id, success)

            # Log
            episode_reward += reward
            episode_gains.append(info['learning_gain'])
            knowledge_trajectory.append(learner.knowledge.copy())

            step_log = {
                'episode': len(self.episode_logs),
                'step': step,
                'topic': topic_id,
                'strategy': int(strategy),
                'reward': reward,
                'learning_gain': info['learning_gain'],
                'engagement': info['engagement'],
                'frustration': info['frustration'],
                'mean_knowledge': info['mean_knowledge'],
            }
            self.step_logs.append(step_log)

            if done:
                break

        # PPO update at end of episode
        ppo_stats = self.pedagogy_agent.update()

        episode_summary = {
            'episode': len(self.episode_logs),
            'total_reward': episode_reward,
            'mean_reward': episode_reward / (step + 1),
            'total_learning_gain': sum(episode_gains),
            'mean_learning_gain': np.mean(episode_gains),
            'final_mean_knowledge': float(np.mean(learner.knowledge)),
            'final_engagement': float(learner.engagement),
            'final_frustration': float(learner.frustration),
            'steps': step + 1,
            'mastered': self.env.is_mastered(learner),
            'learner_type': int(learner.learner_type),
            'knowledge_trajectory': knowledge_trajectory,
            'ppo_stats': ppo_stats,
        }
        self.episode_logs.append(episode_summary)

        return episode_summary

    def _get_available_topics(self, learner: LearnerProfile) -> List[int]:
        """Get topics where prerequisites are met."""
        available = []
        for topic in self.env.topics:
            prereq_met = all(
                learner.knowledge[p] >= 0.3 for p in topic.prerequisites
            )
            not_mastered = learner.knowledge[topic.topic_id] < 0.9
            if prereq_met and not_mastered:
                available.append(topic.topic_id)
        return available

    def train(self, n_episodes: int = 200, max_steps: int = 50,
              log_interval: int = 20) -> List[Dict]:
        """Train all agents over multiple episodes."""
        print(f"{'='*60}")
        print(f"ATLAS Training: {n_episodes} episodes")
        print(f"{'='*60}")

        all_summaries = []
        for ep in range(n_episodes):
            # Vary learner types for robustness
            learner_type = LearnerType(ep % len(LearnerType))
            learner = self.env.create_learner(learner_type)

            summary = self.run_episode(learner, max_steps)
            all_summaries.append(summary)

            if (ep + 1) % log_interval == 0:
                recent = all_summaries[-log_interval:]
                avg_reward = np.mean([s['total_reward'] for s in recent])
                avg_gain = np.mean([s['total_learning_gain'] for s in recent])
                avg_knowledge = np.mean([s['final_mean_knowledge'] for s in recent])
                mastery_rate = np.mean([s['mastered'] for s in recent])

                print(f"Episode {ep+1:4d} | "
                      f"Reward: {avg_reward:7.2f} | "
                      f"Learn: {avg_gain:.3f} | "
                      f"Knowledge: {avg_knowledge:.3f} | "
                      f"Mastery: {mastery_rate:.1%}")

        return all_summaries

    def evaluate(self, n_episodes: int = 50, max_steps: int = 50) -> Dict:
        """Evaluate current policy without learning."""
        results = {ltype.name: [] for ltype in LearnerType}

        for ep in range(n_episodes):
            learner_type = LearnerType(ep % len(LearnerType))
            learner = self.env.create_learner(learner_type)

            # Run without updating
            episode_reward = 0
            episode_gains = []

            for step in range(max_steps):
                state = self.env.get_learner_state(learner)
                ctx = self.env.get_context_for_bandit(learner)
                available = self._get_available_topics(learner)
                if not available:
                    available = list(range(min(3, self.config.n_topics)))

                topic_id, _ = self.curriculum_agent.select_arm(ctx, available)
                action, _ = self.pedagogy_agent.get_action(state, deterministic=True)
                strategy = TeachingStrategy(action)

                reward, info = self.env.step(learner, topic_id, strategy)
                episode_reward += reward
                episode_gains.append(info['learning_gain'])

                if self.env.is_mastered(learner):
                    break

            results[learner_type.name].append({
                'total_reward': episode_reward,
                'total_gain': sum(episode_gains),
                'final_knowledge': float(np.mean(learner.knowledge)),
                'mastered': self.env.is_mastered(learner),
                'steps': step + 1,
            })

        return results


class RandomBaseline:
    """Random baseline for comparison."""

    def __init__(self, n_topics: int, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.n_topics = n_topics

    def select_topic(self, available: List[int]) -> int:
        return self.rng.choice(available)

    def select_strategy(self) -> TeachingStrategy:
        return TeachingStrategy(self.rng.randint(0, len(TeachingStrategy)))


class FixedBaseline:
    """Fixed curriculum baseline (sequential topics, lecture-only)."""

    def __init__(self, n_topics: int):
        self.n_topics = n_topics
        self.current_topic = 0

    def select_topic(self, learner: LearnerProfile) -> int:
        for i in range(self.n_topics):
            if learner.knowledge[i] < 0.8:
                return i
        return self.n_topics - 1

    def select_strategy(self) -> TeachingStrategy:
        return TeachingStrategy.LECTURE


class OracleBaseline:
    """Oracle baseline that always picks the optimal topic/strategy."""

    def __init__(self, env: TutorialEnvironment):
        self.env = env

    def select_topic(self, learner: LearnerProfile) -> int:
        return self.env.get_optimal_topic(learner)

    def select_strategy(self, learner: LearnerProfile) -> TeachingStrategy:
        return self.env.get_optimal_strategy(learner)
