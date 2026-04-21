"""
Test Suite for ATLAS
=====================
Tests for PPO, LinUCB, environment, and orchestrator.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from src.rl.ppo import PPOAgent, PPOConfig, NeuralNetwork
from src.rl.contextual_bandit import LinUCBAgent, LinUCBConfig, HybridBandit
from src.environment.tutorial_env import (
    TutorialEnvironment, TeachingStrategy, LearnerType
)
from src.agents.orchestrator import (
    ATLASOrchestrator, OrchestratorConfig,
    AssessmentAgent, DifficultyAgent,
    RandomBaseline, FixedBaseline, OracleBaseline,
)


class TestNeuralNetwork(unittest.TestCase):
    def test_forward_shape(self):
        net = NeuralNetwork([16, 32, 8])
        x = np.random.randn(1, 16)
        out, activations = net.forward(x)
        self.assertEqual(out.shape, (1, 8))

    def test_params_roundtrip(self):
        net = NeuralNetwork([4, 8, 2])
        params = net.get_params()
        net2 = NeuralNetwork([4, 8, 2], seed=999)
        net2.set_params(params)
        x = np.random.randn(1, 4)
        out1, _ = net.forward(x)
        out2, _ = net2.forward(x)
        np.testing.assert_array_almost_equal(out1, out2)


class TestPPO(unittest.TestCase):
    def setUp(self):
        self.config = PPOConfig(state_dim=8, action_dim=4, hidden_dim=16, batch_size=8)
        self.agent = PPOAgent(self.config)

    def test_get_action(self):
        state = np.random.randn(8)
        action, log_prob = self.agent.get_action(state)
        self.assertIn(action, range(4))
        self.assertTrue(log_prob <= 0)

    def test_get_value(self):
        state = np.random.randn(8)
        value = self.agent.get_value(state)
        self.assertIsInstance(value, float)

    def test_store_and_update(self):
        for _ in range(20):
            s = np.random.randn(8)
            a, lp = self.agent.get_action(s)
            ns = np.random.randn(8)
            self.agent.store_transition(s, a, 1.0, ns, False, lp)

        stats = self.agent.update()
        self.assertIn('policy_loss', stats)
        self.assertIn('value_loss', stats)

    def test_deterministic_action(self):
        state = np.random.randn(8)
        actions = [self.agent.get_action(state, deterministic=True)[0] for _ in range(10)]
        self.assertTrue(all(a == actions[0] for a in actions))

    def test_gae_computation(self):
        rewards = np.array([1.0, 0.5, 0.8])
        values = np.array([0.5, 0.3, 0.6])
        next_values = np.array([0.3, 0.6, 0.0])
        dones = np.array([0, 0, 1])
        advantages, returns = self.agent.compute_gae(rewards, values, next_values, dones)
        self.assertEqual(len(advantages), 3)
        self.assertEqual(len(returns), 3)


class TestLinUCB(unittest.TestCase):
    def setUp(self):
        self.config = LinUCBConfig(context_dim=8, n_arms=5, alpha=1.0)
        self.agent = LinUCBAgent(self.config)

    def test_select_arm(self):
        context = np.random.randn(8)
        arm, info = self.agent.select_arm(context)
        self.assertIn(arm, range(5))
        self.assertIn('ucb_scores', info)

    def test_update(self):
        context = np.random.randn(8)
        self.agent.update(0, context, 1.0)
        self.assertEqual(self.agent.arm_counts[0], 1)
        self.assertEqual(self.agent.total_steps, 1)

    def test_exploration_decay(self):
        initial_alpha = self.agent.alpha
        for _ in range(100):
            ctx = np.random.randn(8)
            arm, _ = self.agent.select_arm(ctx)
            self.agent.update(arm, ctx, np.random.rand())
        self.assertLess(self.agent.alpha, initial_alpha)

    def test_available_arms(self):
        context = np.random.randn(8)
        arm, _ = self.agent.select_arm(context, available_arms=[1, 3])
        self.assertIn(arm, [1, 3])

    def test_arm_statistics(self):
        for _ in range(10):
            ctx = np.random.randn(8)
            arm, _ = self.agent.select_arm(ctx)
            self.agent.update(arm, ctx, 1.0)
        stats = self.agent.get_arm_statistics()
        self.assertEqual(len(stats), 5)


class TestHybridBandit(unittest.TestCase):
    def test_hybrid_select(self):
        config = LinUCBConfig(context_dim=8, n_arms=5)
        bandit = HybridBandit(config, arm_feature_dim=4)
        ctx = np.random.randn(8)
        arm_feats = {i: np.random.randn(4) for i in range(5)}
        arm, info = bandit.select_arm_hybrid(ctx, arm_feats)
        self.assertIn(arm, range(5))

    def test_hybrid_update(self):
        config = LinUCBConfig(context_dim=8, n_arms=5)
        bandit = HybridBandit(config, arm_feature_dim=4)
        ctx = np.random.randn(8)
        arm_feat = np.random.randn(4)
        bandit.update_hybrid(0, ctx, arm_feat, 1.0)
        self.assertEqual(bandit.arm_counts[0], 1)


class TestTutorialEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = TutorialEnvironment(n_topics=10, seed=42)

    def test_create_learner(self):
        learner = self.env.create_learner()
        self.assertEqual(len(learner.knowledge), 10)
        self.assertTrue(0 <= learner.engagement <= 1)

    def test_step(self):
        learner = self.env.create_learner(LearnerType.FAST_VISUAL)
        reward, info = self.env.step(learner, 0, TeachingStrategy.LECTURE)
        self.assertIsInstance(reward, float)
        self.assertIn('learning_gain', info)
        self.assertGreaterEqual(info['learning_gain'], 0)

    def test_state_shape(self):
        learner = self.env.create_learner()
        state = self.env.get_learner_state(learner)
        self.assertEqual(state.shape, (32,))

    def test_bandit_context_shape(self):
        learner = self.env.create_learner()
        ctx = self.env.get_context_for_bandit(learner)
        self.assertEqual(ctx.shape, (16,))

    def test_knowledge_increases(self):
        learner = self.env.create_learner(LearnerType.FAST_VISUAL)
        learner.engagement = 1.0
        learner.knowledge = np.zeros(10)
        initial = learner.knowledge[0]
        for _ in range(20):
            self.env.step(learner, 0, TeachingStrategy.INTERACTIVE_SIM)
        self.assertGreater(learner.knowledge[0], initial)

    def test_mastery_check(self):
        learner = self.env.create_learner()
        learner.knowledge = np.ones(10) * 0.9
        self.assertTrue(self.env.is_mastered(learner, threshold=0.8))


class TestAssessmentAgent(unittest.TestCase):
    def test_belief_update(self):
        agent = AssessmentAgent(n_topics=5)
        agent.initialize_learner("test")
        agent.update_belief("test", 0, 0.8)
        est = agent.get_estimated_knowledge("test")
        self.assertEqual(len(est), 5)

    def test_uncertainty_decreases(self):
        agent = AssessmentAgent(n_topics=5)
        agent.initialize_learner("test")
        u1 = agent.get_uncertainty("test")[0]
        for _ in range(10):
            agent.update_belief("test", 0, 0.7)
        u2 = agent.get_uncertainty("test")[0]
        self.assertLess(u2, u1)


class TestDifficultyAgent(unittest.TestCase):
    def test_adjustment(self):
        agent = DifficultyAgent()
        adj = agent.get_difficulty_adjustment("test", [0.9, 0.9, 0.9])
        self.assertGreater(adj, 0)  # Should increase difficulty

    def test_strategy_adjustment(self):
        agent = DifficultyAgent()
        result = agent.adjust_strategy(TeachingStrategy.PRACTICE_EASY, 0.2)
        self.assertEqual(result, TeachingStrategy.PRACTICE_MEDIUM)


class TestOrchestrator(unittest.TestCase):
    def test_single_episode(self):
        config = OrchestratorConfig(n_topics=5, seed=42)
        config.ppo_config = PPOConfig(state_dim=32, action_dim=8, hidden_dim=32, batch_size=8, seed=42)
        config.bandit_config = LinUCBConfig(context_dim=16, n_arms=5, seed=142)
        orch = ATLASOrchestrator(config)
        summary = orch.run_episode(max_steps=20)
        self.assertIn('total_reward', summary)
        self.assertIn('final_mean_knowledge', summary)
        self.assertGreater(summary['steps'], 0)

    def test_training(self):
        config = OrchestratorConfig(n_topics=5, seed=42)
        config.ppo_config = PPOConfig(state_dim=32, action_dim=8, hidden_dim=32, batch_size=8, seed=42)
        config.bandit_config = LinUCBConfig(context_dim=16, n_arms=5, seed=142)
        orch = ATLASOrchestrator(config)
        summaries = orch.train(n_episodes=10, max_steps=15, log_interval=5)
        self.assertEqual(len(summaries), 10)


class TestBaselines(unittest.TestCase):
    def test_random_baseline(self):
        bl = RandomBaseline(5, seed=42)
        topic = bl.select_topic([0, 1, 2])
        self.assertIn(topic, [0, 1, 2])
        strategy = bl.select_strategy()
        self.assertIsInstance(strategy, TeachingStrategy)

    def test_fixed_baseline(self):
        env = TutorialEnvironment(n_topics=5, seed=42)
        bl = FixedBaseline(5)
        learner = env.create_learner()
        topic = bl.select_topic(learner)
        self.assertIn(topic, range(5))

    def test_oracle_baseline(self):
        env = TutorialEnvironment(n_topics=5, seed=42)
        bl = OracleBaseline(env)
        learner = env.create_learner()
        topic = bl.select_topic(learner)
        strategy = bl.select_strategy(learner)
        self.assertIn(topic, range(5))
        self.assertIsInstance(strategy, TeachingStrategy)


if __name__ == '__main__':
    unittest.main(verbosity=2)
