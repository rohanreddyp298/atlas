"""
Tutorial Environment with Simulated Learners
=============================================
Provides a realistic simulation of learner behavior for training
and evaluating the ATLAS tutorial agents.

Learner Model:
- Knowledge state across multiple topics (continuous 0-1)
- Engagement level (dynamic, affected by teaching quality)
- Learning style preferences (visual, textual, interactive, example-based)
- Forgetting dynamics (exponential decay)
- Zone of Proximal Development (optimal difficulty window)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import IntEnum


class TeachingStrategy(IntEnum):
    """Available teaching strategies (PPO action space)."""
    LECTURE = 0           # Conceptual explanation
    WORKED_EXAMPLE = 1    # Step-by-step worked example
    PRACTICE_EASY = 2     # Easy practice problems
    PRACTICE_MEDIUM = 3   # Medium practice problems
    PRACTICE_HARD = 4     # Hard practice problems
    INTERACTIVE_SIM = 5   # Interactive simulation
    PEER_DISCUSSION = 6   # Peer discussion prompt
    REVIEW_RECAP = 7      # Review and recap


class LearnerType(IntEnum):
    """Learner archetype types."""
    FAST_VISUAL = 0       # Quick learner, prefers visual/interactive
    SLOW_METHODICAL = 1   # Slower but thorough, prefers examples
    SOCIAL_LEARNER = 2    # Learns best through discussion
    PRACTICE_ORIENTED = 3 # Learns by doing


@dataclass
class LearnerProfile:
    """Represents a simulated learner."""
    learner_id: str
    learner_type: LearnerType
    knowledge: np.ndarray        # Knowledge per topic [0, 1]
    engagement: float = 0.7      # Current engagement [0, 1]
    frustration: float = 0.0     # Frustration level [0, 1]
    learning_rate: float = 0.1   # Base learning rate
    style_preferences: np.ndarray = None  # Strategy preferences
    session_history: List[Dict] = field(default_factory=list)
    total_interactions: int = 0


@dataclass
class TopicInfo:
    """Information about a curriculum topic."""
    topic_id: int
    name: str
    difficulty: float           # Inherent difficulty [0, 1]
    prerequisites: List[int]    # Required prerequisite topic IDs
    feature_vector: np.ndarray = None  # For hybrid bandit


class TutorialEnvironment:
    """
    Simulated tutorial environment for RL training.
    
    Models realistic learner dynamics including:
    - Knowledge acquisition with prerequisite dependencies
    - Engagement fluctuation based on difficulty match
    - Forgetting (spaced repetition effects)
    - Zone of Proximal Development
    """

    def __init__(self, n_topics: int = 10, n_learner_types: int = 4, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.n_topics = n_topics
        self.n_strategies = len(TeachingStrategy)

        # Create curriculum
        self.topics = self._create_curriculum(n_topics)

        # Strategy effectiveness per learner type
        # Shape: (n_learner_types, n_strategies)
        # Highly polarized: right strategy = 1.0, wrong = 0.1-0.3
        self.strategy_effectiveness = np.array([
            # LECT  WORK  EASY  MED   HARD  INTER PEER  REV
            [0.3,  0.4,  0.15, 0.3,  0.2,  1.0,  0.25, 0.35],  # FAST_VISUAL
            [0.5,  1.0,  0.4,  0.5,  0.15, 0.25, 0.2,  0.6],   # SLOW_METHODICAL
            [0.2,  0.3,  0.2,  0.3,  0.15, 0.35, 1.0,  0.25],  # SOCIAL_LEARNER
            [0.15, 0.35, 0.5,  1.0,  0.4,  0.45, 0.2,  0.3],   # PRACTICE_ORIENTED
        ])

        # Statistics
        self.episode_count = 0
        self.step_count = 0

    def _create_curriculum(self, n_topics: int) -> List[TopicInfo]:
        """Create a structured curriculum with prerequisites."""
        topic_names = [
            "Variables & Types", "Control Flow", "Functions",
            "Data Structures", "OOP Basics", "File I/O",
            "Error Handling", "Algorithms", "Testing", "Design Patterns"
        ]

        topics = []
        for i in range(min(n_topics, len(topic_names))):
            prereqs = []
            if i >= 2:
                prereqs.append(i - 2)
            if i >= 1:
                prereqs.append(i - 1)

            feature = self.rng.randn(8)
            feature[0] = i / n_topics  # Difficulty ordering
            feature[1] = len(prereqs) / 3  # Prerequisite density

            topics.append(TopicInfo(
                topic_id=i,
                name=topic_names[i] if i < len(topic_names) else f"Topic_{i}",
                difficulty=0.1 + 0.6 * (i / max(n_topics - 1, 1)),
                prerequisites=prereqs,
                feature_vector=feature,
            ))
        return topics

    def create_learner(self, learner_type: Optional[LearnerType] = None) -> LearnerProfile:
        """Create a new simulated learner."""
        if learner_type is None:
            learner_type = LearnerType(self.rng.randint(0, len(LearnerType)))

        # Initialize with some prior knowledge on early topics
        knowledge = np.zeros(self.n_topics)
        knowledge[0] = self.rng.uniform(0.4, 0.7)
        if self.n_topics > 1:
            knowledge[1] = self.rng.uniform(0.1, 0.4)

        # Learning rate varies by type
        lr_map = {
            LearnerType.FAST_VISUAL: 0.45,
            LearnerType.SLOW_METHODICAL: 0.28,
            LearnerType.SOCIAL_LEARNER: 0.31,
            LearnerType.PRACTICE_ORIENTED: 0.39,
        }

        return LearnerProfile(
            learner_id=f"learner_{self.rng.randint(10000)}",
            learner_type=learner_type,
            knowledge=knowledge,
            engagement=self.rng.uniform(0.7, 0.95),
            learning_rate=lr_map[learner_type],
            style_preferences=self.strategy_effectiveness[learner_type].copy(),
        )

    def get_learner_state(self, learner: LearnerProfile) -> np.ndarray:
        """
        Extract state vector from learner profile.
        
        State vector (dim=32):
        - Knowledge levels per topic (10)
        - Engagement (1)
        - Frustration (1)
        - Learner type one-hot (4)
        - Recent performance (5)
        - Topic difficulty gap (5)
        - Prerequisite readiness (5)
        - Session length (1)
        """
        state = np.zeros(32)

        # Knowledge levels (first n_topics slots)
        state[:self.n_topics] = learner.knowledge[:self.n_topics]

        # Engagement and frustration
        state[10] = learner.engagement
        state[11] = learner.frustration

        # Learner type one-hot
        state[12 + learner.learner_type] = 1.0

        # Recent performance (last 5 interactions)
        recent = learner.session_history[-5:]
        for i, interaction in enumerate(recent):
            state[16 + i] = interaction.get('learning_gain', 0)

        # Topic difficulty gap (up to 5 topics)
        for i in range(min(5, self.n_topics)):
            state[21 + i] = self.topics[i].difficulty - learner.knowledge[i]

        # Prerequisite readiness for topics
        for i in range(min(5, self.n_topics)):
            topic = self.topics[i]
            if topic.prerequisites:
                state[26 + i] = min(learner.knowledge[p] for p in topic.prerequisites)
            else:
                state[26 + i] = 1.0

        # Session length (normalized)
        state[31] = min(learner.total_interactions / 80, 1.0)

        return state

    def get_context_for_bandit(self, learner: LearnerProfile) -> np.ndarray:
        """
        Extract context vector for the curriculum bandit.
        
        Context (dim=16):
        - Knowledge summary stats (4)
        - Engagement & frustration (2)
        - Learner type one-hot (4)
        - Recent trend (3)
        - Session info (3)
        """
        ctx = np.zeros(16)

        # Knowledge summary
        ctx[0] = np.mean(learner.knowledge)
        ctx[1] = np.std(learner.knowledge)
        ctx[2] = np.min(learner.knowledge)
        ctx[3] = np.max(learner.knowledge)

        # Engagement
        ctx[4] = learner.engagement
        ctx[5] = learner.frustration

        # Type
        ctx[6 + learner.learner_type] = 1.0

        # Recent trend
        if len(learner.session_history) >= 3:
            recent_gains = [h.get('learning_gain', 0) for h in learner.session_history[-3:]]
            ctx[10] = np.mean(recent_gains)
            ctx[11] = recent_gains[-1] - recent_gains[0]  # Trend
            ctx[12] = 1.0 if all(g > 0 for g in recent_gains) else 0.0

        # Session info
        ctx[13] = min(learner.total_interactions / 80, 1.0)
        ctx[14] = len(learner.session_history) / max(50, 1)
        ctx[15] = learner.learning_rate

        return ctx

    def step(self, learner: LearnerProfile, topic_id: int,
             strategy: TeachingStrategy) -> Tuple[float, Dict]:
        """
        Execute one teaching step.
        
        Returns:
            reward: Combined learning gain + engagement reward
            info: Detailed information about the interaction
        """
        topic = self.topics[topic_id]

        # 1. Check prerequisite readiness (sharp threshold)
        prereq_readiness = 1.0
        if topic.prerequisites:
            min_prereq = min(learner.knowledge[p] for p in topic.prerequisites)
            # Sharp sigmoid: prerequisites below 0.4 severely penalize learning
            prereq_readiness = 1.0 / (1.0 + np.exp(-12 * (min_prereq - 0.4)))

        # 2. Compute difficulty match (Zone of Proximal Development)
        current_knowledge = learner.knowledge[topic_id]
        difficulty_gap = topic.difficulty - current_knowledge
        zpd_match = np.exp(-2 * (difficulty_gap - 0.3) ** 2)  # Optimal gap ~ 0.3

        # 3. Strategy effectiveness for this learner
        base_effectiveness = self.strategy_effectiveness[learner.learner_type][strategy]

        # 4. Compute learning gain
        learning_gain = (
            learner.learning_rate
            * base_effectiveness
            * zpd_match
            * prereq_readiness
            * learner.engagement
            * (1.0 - 0.5 * learner.frustration)
            * (1.0 + 0.1 * self.rng.randn())  # Noise
        )
        learning_gain = np.clip(learning_gain, 0, 0.35)

        # 5. Update knowledge
        old_knowledge = learner.knowledge[topic_id]
        learner.knowledge[topic_id] = min(1.0, old_knowledge + learning_gain)

        # 6. Apply forgetting to other topics
        for i in range(self.n_topics):
            if i != topic_id and learner.knowledge[i] > 0.01:
                forget_rate = 0.002 * (1 - learner.knowledge[i] ** 2)
                learner.knowledge[i] = max(0, learner.knowledge[i] - forget_rate)

        # 7. Update engagement
        engagement_delta = 0.0
        if difficulty_gap > 0.6:  # Too hard
            engagement_delta = -0.04
            learner.frustration = min(1.0, learner.frustration + 0.05)
        elif difficulty_gap < -0.2:  # Too easy
            engagement_delta = -0.02
        else:  # Good match
            engagement_delta = 0.05
            learner.frustration = max(0, learner.frustration - 0.08)

        # Strategy variety bonus
        if len(learner.session_history) >= 3:
            recent_strategies = [h['strategy'] for h in learner.session_history[-3:]]
            if strategy not in recent_strategies:
                engagement_delta += 0.02

        learner.engagement = np.clip(learner.engagement + engagement_delta, 0.1, 1.0)

        # 8. Compute reward
        reward = (
            3.0 * learning_gain           # Primary: actual learning
            + 0.5 * learner.engagement    # Secondary: keep engaged
            - 0.3 * learner.frustration   # Penalty: frustration
            + 0.2 * prereq_readiness      # Bonus: good sequencing
        )

        # 9. Record history
        interaction = {
            'step': self.step_count,
            'topic': topic_id,
            'strategy': int(strategy),
            'learning_gain': float(learning_gain),
            'engagement': float(learner.engagement),
            'frustration': float(learner.frustration),
            'reward': float(reward),
            'zpd_match': float(zpd_match),
            'prereq_readiness': float(prereq_readiness),
        }
        learner.session_history.append(interaction)
        learner.total_interactions += 1
        self.step_count += 1

        info = {
            **interaction,
            'knowledge_before': float(old_knowledge),
            'knowledge_after': float(learner.knowledge[topic_id]),
            'mean_knowledge': float(np.mean(learner.knowledge)),
        }

        return reward, info

    def is_mastered(self, learner: LearnerProfile, threshold: float = 0.7) -> bool:
        """Check if learner has mastered all topics."""
        return np.all(learner.knowledge >= threshold)

    def get_optimal_topic(self, learner: LearnerProfile) -> int:
        """Get the theoretically optimal next topic (oracle)."""
        best_topic = 0
        best_score = -np.inf

        for i, topic in enumerate(self.topics):
            if learner.knowledge[i] >= 0.85:
                continue  # Already mastered

            prereq_ready = all(
                learner.knowledge[p] >= 0.3 for p in topic.prerequisites
            )
            if not prereq_ready:
                continue

            # ZPD score: prefer topics with moderate gap
            gap = topic.difficulty - learner.knowledge[i]
            zpd = np.exp(-2 * (gap - 0.3) ** 2)
            # Prioritize topics with lower knowledge (most room to grow)
            score = zpd * (1 - learner.knowledge[i])
            if score > best_score:
                best_score = score
                best_topic = i

        return best_topic

    def get_optimal_strategy(self, learner: LearnerProfile) -> TeachingStrategy:
        """Get the theoretically optimal strategy (oracle with variety)."""
        # Pick best strategy for this learner type
        prefs = self.strategy_effectiveness[learner.learner_type]
        # Add variety: cycle through top-3 strategies
        top3 = np.argsort(prefs)[-3:]
        step = learner.total_interactions % 3
        return TeachingStrategy(top3[step])