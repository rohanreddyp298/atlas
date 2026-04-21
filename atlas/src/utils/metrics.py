"""
Metrics and Visualization Utilities
=====================================
Performance metrics, learning curves, and publication-quality plots.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json


def compute_learning_metrics(episode_logs: List[Dict]) -> Dict:
    """Compute comprehensive metrics from episode logs."""
    if not episode_logs:
        return {}

    rewards = [e['total_reward'] for e in episode_logs]
    gains = [e['total_learning_gain'] for e in episode_logs]
    knowledge = [e['final_mean_knowledge'] for e in episode_logs]
    mastery = [e['mastered'] for e in episode_logs]
    engagement = [e['final_engagement'] for e in episode_logs]

    def moving_average(data, window=20):
        if len(data) < window:
            return data
        return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]

    return {
        'reward': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'max': float(np.max(rewards)),
            'min': float(np.min(rewards)),
            'moving_avg': moving_average(rewards),
            'raw': rewards,
        },
        'learning_gain': {
            'mean': float(np.mean(gains)),
            'std': float(np.std(gains)),
            'moving_avg': moving_average(gains),
            'raw': gains,
        },
        'knowledge': {
            'mean': float(np.mean(knowledge)),
            'final_20_mean': float(np.mean(knowledge[-20:])) if len(knowledge) >= 20 else float(np.mean(knowledge)),
            'moving_avg': moving_average(knowledge),
            'raw': knowledge,
        },
        'mastery_rate': {
            'overall': float(np.mean(mastery)),
            'final_20': float(np.mean(mastery[-20:])) if len(mastery) >= 20 else float(np.mean(mastery)),
            'moving_avg': moving_average([float(m) for m in mastery]),
        },
        'engagement': {
            'mean': float(np.mean(engagement)),
            'moving_avg': moving_average(engagement),
        },
        'convergence': {
            'episodes_to_50pct_mastery': _find_threshold_episode(mastery, 0.5),
            'episodes_to_80pct_mastery': _find_threshold_episode(mastery, 0.8),
        }
    }


def _find_threshold_episode(mastery: List, threshold: float, window: int = 20) -> Optional[int]:
    """Find first episode where moving mastery rate exceeds threshold."""
    for i in range(window, len(mastery)):
        rate = np.mean(mastery[max(0, i-window):i+1])
        if rate >= threshold:
            return i
    return None


def compute_baseline_comparison(atlas_metrics: Dict,
                                 random_metrics: Dict,
                                 fixed_metrics: Dict,
                                 oracle_metrics: Dict = None) -> Dict:
    """Compare ATLAS against baselines."""
    comparison = {
        'ATLAS': {
            'mean_reward': atlas_metrics['reward']['mean'],
            'mean_knowledge': atlas_metrics['knowledge']['mean'],
            'mastery_rate': atlas_metrics['mastery_rate']['overall'],
        },
        'Random': {
            'mean_reward': random_metrics['reward']['mean'],
            'mean_knowledge': random_metrics['knowledge']['mean'],
            'mastery_rate': random_metrics['mastery_rate']['overall'],
        },
        'Fixed': {
            'mean_reward': fixed_metrics['reward']['mean'],
            'mean_knowledge': fixed_metrics['knowledge']['mean'],
            'mastery_rate': fixed_metrics['mastery_rate']['overall'],
        },
    }

    if oracle_metrics:
        comparison['Oracle'] = {
            'mean_reward': oracle_metrics['reward']['mean'],
            'mean_knowledge': oracle_metrics['knowledge']['mean'],
            'mastery_rate': oracle_metrics['mastery_rate']['overall'],
        }

    # Improvement percentages
    comparison['improvement_over_random'] = {
        'reward': (atlas_metrics['reward']['mean'] - random_metrics['reward']['mean'])
                  / max(abs(random_metrics['reward']['mean']), 0.01) * 100,
        'knowledge': (atlas_metrics['knowledge']['mean'] - random_metrics['knowledge']['mean'])
                     / max(random_metrics['knowledge']['mean'], 0.01) * 100,
    }

    comparison['improvement_over_fixed'] = {
        'reward': (atlas_metrics['reward']['mean'] - fixed_metrics['reward']['mean'])
                  / max(abs(fixed_metrics['reward']['mean']), 0.01) * 100,
        'knowledge': (atlas_metrics['knowledge']['mean'] - fixed_metrics['knowledge']['mean'])
                     / max(fixed_metrics['knowledge']['mean'], 0.01) * 100,
    }

    return comparison


def compute_per_learner_type_metrics(evaluation_results: Dict) -> Dict:
    """Analyze performance broken down by learner type."""
    type_metrics = {}
    for ltype, episodes in evaluation_results.items():
        if not episodes:
            continue
        type_metrics[ltype] = {
            'mean_reward': float(np.mean([e['total_reward'] for e in episodes])),
            'mean_knowledge': float(np.mean([e['final_knowledge'] for e in episodes])),
            'mastery_rate': float(np.mean([e['mastered'] for e in episodes])),
            'mean_steps': float(np.mean([e['steps'] for e in episodes])),
        }
    return type_metrics


def export_results(metrics: Dict, comparison: Dict,
                   per_type: Dict, filepath: str):
    """Export all results to JSON."""
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = {
        'metrics': convert(metrics),
        'comparison': convert(comparison),
        'per_learner_type': convert(per_type),
    }

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
