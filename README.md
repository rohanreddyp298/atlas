# ATLAS: Adaptive Tutorial Learning Agent System

**Reinforcement Learning for Agentic AI Systems — Final Project**

**Author:** Rohan Reddy Patlolla
**NUID:** 002059889

ATLAS is a multi-agent reinforcement learning system for adaptive tutoring. It combines Proximal Policy Optimization (PPO) and Contextual Bandits (LinUCB) to optimize both curriculum selection and teaching strategy for individual learners in real time.

---

## Key Results

| Metric | ATLAS | Random | Fixed Curriculum | Oracle |
|--------|-------|--------|-----------------|--------|
| Mean Knowledge | **0.826** | 0.869 | 0.537 | 0.839 |
| Mastery Rate | **69.4%** → **75% (final 20)** | 91.5% | 49.0% | 100.0% |
| Improvement over Fixed | **+31.8% reward, +53.7% knowledge** | — | baseline | — |

ATLAS shows clear **learning improvement over 500 episodes**: mastery rises from 30% (early) to 80% (late training), reaching 75% mastery in the final evaluation window. The system achieves 50% mastery by episode 20 and 80% mastery by episode 239. The PPO agent discovers learner-type-specific strategies, achieving **92.3% mastery for Slow Methodical learners**. Mean engagement is maintained at 82.0%.

---

## Architecture

```
ATLAS Orchestrator (coordination pipeline)
├── Curriculum Agent   — LinUCB Contextual Bandit → selects WHAT to teach
├── Pedagogy Agent     — PPO Policy Gradient      → selects HOW to teach
├── Difficulty Agent   — Adaptive Calibration      → calibrates challenge level
└── Assessment Agent   — Bayesian Estimation       → evaluates understanding
        │
        ▼
Tutorial Environment (simulated learners + knowledge graph)
├── Fast Visual Learner
├── Slow Methodical Learner
├── Social Learner
└── Practice-Oriented Learner
```

## Two RL Implementations

### 1. PPO (Proximal Policy Optimization)
- **Purpose**: Pedagogy Agent learns optimal teaching strategies
- **State**: 32-dim vector (knowledge, engagement, frustration, learner type, history)
- **Actions**: 8 teaching strategies (lecture, worked example, practice, simulation, etc.)
- **Key features**: GAE advantage estimation, clipped surrogate objective, entropy bonus
- **Implementation**: `src/rl/ppo.py`

### 2. LinUCB Contextual Bandit (+ Hybrid variant)
- **Purpose**: Curriculum Agent selects optimal topics
- **Context**: 16-dim learner profile vector
- **Arms**: Available curriculum topics
- **Key features**: Upper Confidence Bounds, exploration decay, hybrid model with topic features
- **Implementation**: `src/rl/contextual_bandit.py`

---

## Project Structure

```
atlas/
├── src/
│   ├── rl/
│   │   ├── ppo.py                  # PPO algorithm
│   │   └── contextual_bandit.py    # LinUCB + Hybrid Bandit
│   ├── agents/
│   │   └── orchestrator.py         # Multi-agent coordinator + all agents
│   ├── environment/
│   │   └── tutorial_env.py         # Learner simulation environment
│   └── utils/
│       └── metrics.py              # Performance metrics
├── experiments/
│   ├── run_experiments.py          # Full experiment pipeline
│   └── generate_plots.py          # Publication-quality figures
├── tests/
│   └── test_all.py                # 29 unit tests
├── report/
│   └── generate_report.py         # PDF report generator
├── results/
│   ├── results.json               # All metrics and comparisons
│   ├── ATLAS_Technical_Report.pdf  # Comprehensive technical report
│   └── figures/                   # 7 publication-quality plots
└── README.md
```

---

## Installation & Setup

```bash
# Clone the repository
git clone <repo-url>
cd atlas

# Install dependencies (Python 3.8+)
pip install numpy matplotlib reportlab

# Run tests (29 tests)
python -m tests.test_all

# Run full experiments (training + baselines + evaluation + plots)
python experiments/run_experiments.py
python experiments/generate_plots.py

# Generate PDF report
python report/generate_report.py
```

**Requirements**: Python 3.8+, NumPy, Matplotlib, ReportLab (for PDF report)

---

## Experimental Design

### Training Protocol
- 500 training episodes × 80 steps per episode
- 5 curriculum topics (Variables & Types through OOP Basics)
- Learner types cycled for balanced exposure
- PPO updates at episode end; bandit updates every step

### Baselines
| Baseline | Topic Selection | Strategy |
|----------|----------------|----------|
| **Random** | Uniform random | Uniform random |
| **Fixed Curriculum** | Sequential | Always Lecture |
| **Oracle (Upper Bound)** | Optimal ZPD match | Optimal per learner type |

### Metrics
- **Mean Knowledge**: Average knowledge across all topics (0-1)
- **Mastery Rate**: Fraction of episodes achieving ≥0.7 on all topics
- **Total Reward**: Composite reward (learning gain + engagement − frustration)
- **Engagement**: Average learner engagement maintained (0-1)

---

## Learner Simulation Model

The environment models realistic educational dynamics:

- **Zone of Proximal Development (ZPD)**: Learning maximized at ~0.3 difficulty gap
- **Forgetting**: Exponential decay for unpracticed topics (spaced repetition)
- **Engagement Dynamics**: Affected by difficulty match and strategy variety
- **4 Learner Archetypes**: Each with different learning rates and strategy preferences
- **Prerequisite Dependencies**: Topics require prior knowledge foundations

---

## Ethical Considerations

- **Fairness**: Per-learner-type analysis reveals disparities requiring attention
- **Transparency**: System recommendations should be explainable
- **Privacy**: Detailed learner models contain sensitive cognitive data
- **Engagement risks**: Reward function weights learning 6x over engagement to prevent edutainment

---

## License

This project is submitted as coursework for Reinforcement Learning for Agentic AI Systems.
