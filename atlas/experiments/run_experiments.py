"""
ATLAS Experiment Runner
========================
Trains ATLAS, runs baselines, evaluates, and generates all results + plots.

Usage:
    python experiments/run_experiments.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.rl.ppo import PPOConfig
from src.rl.contextual_bandit import LinUCBConfig
from src.environment.tutorial_env import (
    TutorialEnvironment, TeachingStrategy, LearnerType
)
from src.agents.orchestrator import (
    ATLASOrchestrator, OrchestratorConfig,
    RandomBaseline, FixedBaseline, OracleBaseline,
)
from src.utils.metrics import (
    compute_learning_metrics, compute_baseline_comparison,
    compute_per_learner_type_metrics, export_results,
)


def run_atlas_training(n_episodes=300, max_steps=50, seed=42):
    """Train the ATLAS system."""
    print("\n" + "=" * 70)
    print("  PHASE 1: ATLAS TRAINING")
    print("=" * 70)

    config = OrchestratorConfig(
        n_topics=5,
        ppo_config=PPOConfig(
            state_dim=32,
            action_dim=len(TeachingStrategy),
            hidden_dim=64,
            learning_rate=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            n_epochs=4,
            batch_size=32,
            seed=seed,
        ),
        bandit_config=LinUCBConfig(
            context_dim=16,
            n_arms=5,
            alpha=1.5,
            alpha_decay=0.999,
            seed=seed + 100,
        ),
        use_hybrid_bandit=True,
        seed=seed,
    )

    orchestrator = ATLASOrchestrator(config)
    start = time.time()
    summaries = orchestrator.train(n_episodes=n_episodes, max_steps=max_steps, log_interval=30)
    training_time = time.time() - start
    print(f"\nTraining completed in {training_time:.1f}s")

    return orchestrator, summaries


def run_baseline(env, baseline, n_episodes=200, max_steps=50, name="Baseline"):
    """Run a baseline agent."""
    print(f"\n  Running {name}...")
    logs = []

    for ep in range(n_episodes):
        learner_type = LearnerType(ep % len(LearnerType))
        learner = env.create_learner(learner_type)
        ep_reward = 0
        ep_gains = []

        for step in range(max_steps):
            available = [i for i in range(env.n_topics)
                         if all(learner.knowledge[p] >= 0.3
                                for p in env.topics[i].prerequisites)
                         and learner.knowledge[i] < 0.9]
            if not available:
                available = list(range(min(3, env.n_topics)))

            if isinstance(baseline, RandomBaseline):
                topic = baseline.select_topic(available)
                strategy = baseline.select_strategy()
            elif isinstance(baseline, FixedBaseline):
                topic = baseline.select_topic(learner)
                strategy = baseline.select_strategy()
            elif isinstance(baseline, OracleBaseline):
                topic = baseline.select_topic(learner)
                strategy = baseline.select_strategy(learner)
            else:
                break

            reward, info = env.step(learner, topic, strategy)
            ep_reward += reward
            ep_gains.append(info['learning_gain'])

            if env.is_mastered(learner):
                break

        logs.append({
            'total_reward': ep_reward,
            'total_learning_gain': sum(ep_gains),
            'mean_learning_gain': float(np.mean(ep_gains)) if ep_gains else 0,
            'final_mean_knowledge': float(np.mean(learner.knowledge)),
            'final_engagement': float(learner.engagement),
            'final_frustration': float(learner.frustration),
            'mastered': env.is_mastered(learner),
            'steps': step + 1,
            'learner_type': int(learner.learner_type),
        })

    return logs


def run_all_baselines(n_episodes=200, max_steps=50, seed=42):
    """Run all baseline comparisons."""
    print("\n" + "=" * 70)
    print("  PHASE 2: BASELINE COMPARISONS")
    print("=" * 70)

    env = TutorialEnvironment(n_topics=5, seed=seed)

    random_logs = run_baseline(
        env, RandomBaseline(5, seed=seed),
        n_episodes, max_steps, "Random Baseline"
    )

    env2 = TutorialEnvironment(n_topics=5, seed=seed)
    fixed_logs = run_baseline(
        env2, FixedBaseline(5),
        n_episodes, max_steps, "Fixed Curriculum Baseline"
    )

    env3 = TutorialEnvironment(n_topics=5, seed=seed)
    oracle_logs = run_baseline(
        env3, OracleBaseline(env3),
        n_episodes, max_steps, "Oracle (Upper Bound)"
    )

    return random_logs, fixed_logs, oracle_logs


def generate_plots_data(atlas_metrics, random_metrics, fixed_metrics,
                        oracle_metrics, per_type_metrics, eval_results,
                        atlas_summaries, output_dir):
    """Generate JSON data for all visualization plots."""

    # 1. Learning curves
    learning_curves = {
        'atlas_reward': atlas_metrics['reward']['moving_avg'],
        'atlas_knowledge': atlas_metrics['knowledge']['moving_avg'],
        'atlas_mastery': atlas_metrics['mastery_rate']['moving_avg'],
        'atlas_engagement': atlas_metrics['engagement']['moving_avg'],
        'atlas_learning_gain': atlas_metrics['learning_gain']['moving_avg'],
    }

    with open(os.path.join(output_dir, 'learning_curves.json'), 'w') as f:
        json.dump(learning_curves, f)

    # 2. Baseline comparison
    comparison_data = {
        'methods': ['Random', 'Fixed\nCurriculum', 'ATLAS\n(Ours)', 'Oracle\n(Upper Bound)'],
        'rewards': [
            random_metrics['reward']['mean'],
            fixed_metrics['reward']['mean'],
            atlas_metrics['reward']['mean'],
            oracle_metrics['reward']['mean'],
        ],
        'knowledge': [
            random_metrics['knowledge']['mean'],
            fixed_metrics['knowledge']['mean'],
            atlas_metrics['knowledge']['mean'],
            oracle_metrics['knowledge']['mean'],
        ],
        'mastery': [
            random_metrics['mastery_rate']['overall'],
            fixed_metrics['mastery_rate']['overall'],
            atlas_metrics['mastery_rate']['overall'],
            oracle_metrics['mastery_rate']['overall'],
        ],
    }

    with open(os.path.join(output_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison_data, f)

    # 3. Per learner type
    with open(os.path.join(output_dir, 'per_type.json'), 'w') as f:
        json.dump(per_type_metrics, f)

    # 4. Topic selection distribution from bandit
    topic_dist = {}
    for s in atlas_summaries:
        for step_log_idx in range(len(atlas_summaries)):
            pass  # We'll compute from step logs

    # 5. Strategy distribution
    strategy_counts = np.zeros(len(TeachingStrategy))
    for s in atlas_summaries:
        if 'knowledge_trajectory' in s:
            pass

    # 6. Knowledge trajectories (sample from last 10 episodes)
    trajectories = []
    for s in atlas_summaries[-10:]:
        if 'knowledge_trajectory' in s:
            traj = [float(np.mean(k)) for k in s['knowledge_trajectory']]
            trajectories.append(traj)

    with open(os.path.join(output_dir, 'trajectories.json'), 'w') as f:
        json.dump(trajectories, f)

    print(f"\n  Plot data saved to {output_dir}/")


def print_final_report(atlas_metrics, comparison, per_type):
    """Print summary report to console."""
    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  ATLAS Performance:")
    print(f"    Mean Reward:         {atlas_metrics['reward']['mean']:.3f} ± {atlas_metrics['reward']['std']:.3f}")
    print(f"    Mean Knowledge:      {atlas_metrics['knowledge']['mean']:.3f}")
    print(f"    Final 20 Knowledge:  {atlas_metrics['knowledge']['final_20_mean']:.3f}")
    print(f"    Mastery Rate:        {atlas_metrics['mastery_rate']['overall']:.1%}")
    print(f"    Final 20 Mastery:    {atlas_metrics['mastery_rate']['final_20']:.1%}")
    print(f"    Mean Learning Gain:  {atlas_metrics['learning_gain']['mean']:.4f}")
    print(f"    Mean Engagement:     {atlas_metrics['engagement']['mean']:.3f}")

    conv = atlas_metrics['convergence']
    if conv['episodes_to_50pct_mastery']:
        print(f"    50% Mastery at Ep:   {conv['episodes_to_50pct_mastery']}")
    if conv['episodes_to_80pct_mastery']:
        print(f"    80% Mastery at Ep:   {conv['episodes_to_80pct_mastery']}")

    print(f"\n  Improvement over Random:")
    print(f"    Reward:    +{comparison['improvement_over_random']['reward']:.1f}%")
    print(f"    Knowledge: +{comparison['improvement_over_random']['knowledge']:.1f}%")

    print(f"\n  Improvement over Fixed:")
    print(f"    Reward:    +{comparison['improvement_over_fixed']['reward']:.1f}%")
    print(f"    Knowledge: +{comparison['improvement_over_fixed']['knowledge']:.1f}%")

    print(f"\n  Per Learner Type:")
    for ltype, m in per_type.items():
        print(f"    {ltype:20s}: Knowledge={m['mean_knowledge']:.3f}  Mastery={m['mastery_rate']:.1%}")


def main():
    print("=" * 70)
    print("  ATLAS: Adaptive Tutorial Learning Agent System")
    print("  Reinforcement Learning for Agentic AI - Experiments")
    print("=" * 70)

    SEED = 42
    N_TRAIN = 500
    N_BASELINE = 200
    MAX_STEPS = 80

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Train ATLAS
    orchestrator, atlas_summaries = run_atlas_training(N_TRAIN, MAX_STEPS, SEED)

    # Phase 2: Run baselines
    random_logs, fixed_logs, oracle_logs = run_all_baselines(N_BASELINE, MAX_STEPS, SEED)

    # Phase 3: Evaluate ATLAS
    print("\n" + "=" * 70)
    print("  PHASE 3: EVALUATION")
    print("=" * 70)
    eval_results = orchestrator.evaluate(n_episodes=50, max_steps=MAX_STEPS)
    per_type = compute_per_learner_type_metrics(eval_results)

    # Phase 4: Compute metrics
    print("\n  Computing metrics...")
    atlas_metrics = compute_learning_metrics(atlas_summaries)
    random_metrics = compute_learning_metrics(random_logs)
    fixed_metrics = compute_learning_metrics(fixed_logs)
    oracle_metrics = compute_learning_metrics(oracle_logs)

    comparison = compute_baseline_comparison(
        atlas_metrics, random_metrics, fixed_metrics, oracle_metrics
    )

    # Phase 5: Save results
    export_results(atlas_metrics, comparison, per_type,
                   os.path.join(output_dir, 'results.json'))

    generate_plots_data(
        atlas_metrics, random_metrics, fixed_metrics, oracle_metrics,
        per_type, eval_results, atlas_summaries, output_dir
    )

    # Phase 6: Print report
    print_final_report(atlas_metrics, comparison, per_type)

    print(f"\n  All results saved to {output_dir}/")
    print("=" * 70)

    return atlas_metrics, comparison, per_type, atlas_summaries, \
           random_metrics, fixed_metrics, oracle_metrics


if __name__ == '__main__':
    main()
