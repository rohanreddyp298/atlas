"""
Generate all publication-quality plots for the ATLAS report.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

C = {
    'atlas': '#2563eb', 'random': '#9ca3af', 'fixed': '#f59e0b',
    'oracle': '#10b981', 'accent': '#ef4444', 'knowledge': '#6366f1',
    'engagement': '#ec4899', 'mastery': '#14b8a6',
}

def load_results(d):
    with open(os.path.join(d, 'results.json')) as f:
        return json.load(f)

def load_json(d, n):
    p = os.path.join(d, n)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def plot_learning_curves(m, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('ATLAS Training Dynamics Over 500 Episodes', fontsize=15, fontweight='bold', y=1.02)
    for ax, key, title, yl, col in [
        (axes[0,0], 'reward', 'Episode Reward', 'Total Reward', C['atlas']),
        (axes[0,1], 'knowledge', 'Mean Knowledge Level', 'Knowledge [0-1]', C['knowledge']),
        (axes[1,0], 'mastery_rate', 'Mastery Rate', 'Mastery Rate', C['mastery']),
        (axes[1,1], 'learning_gain', 'Total Learning Gain per Episode', 'Cumulative Gain', C['engagement']),
    ]:
        d = m[key]
        ma = d['moving_avg']
        if 'raw' in d:
            ax.plot(d['raw'], alpha=0.12, color=col, linewidth=0.7)
        ax.plot(ma, color=col, linewidth=2, label='Moving Avg (20)')
        if key == 'knowledge':
            ax.axhline(y=0.8, color=C['accent'], linestyle='--', alpha=0.7, label='Mastery Threshold')
            ax.set_ylim(0, 1.05)
        if key == 'mastery_rate':
            ax.fill_between(range(len(ma)), 0, ma, alpha=0.15, color=col)
            ax.set_ylim(0, 1.05)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel(yl)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'learning_curves.png'), dpi=150)
    plt.close()

def plot_baseline_comparison(comp, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('ATLAS vs Baselines: Comparative Analysis', fontsize=15, fontweight='bold', y=1.02)
    methods_k = ['Random', 'Fixed', 'ATLAS', 'Oracle']
    labels = ['Random', 'Fixed\nCurriculum', 'ATLAS\n(Ours)', 'Oracle\n(Upper Bound)']
    colors = [C['random'], C['fixed'], C['atlas'], C['oracle']]
    for ax, metric, title, yl in [
        (axes[0], 'mean_reward', 'Mean Episode Reward', 'Reward'),
        (axes[1], 'mean_knowledge', 'Final Knowledge Level', 'Knowledge [0-1]'),
        (axes[2], 'mastery_rate', 'Mastery Rate', 'Rate'),
    ]:
        vals = [comp[k][metric] for k in methods_k]
        bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=1.5, width=0.65)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(yl)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02*max(abs(x) for x in vals),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        bars[2].set_edgecolor(C['atlas']); bars[2].set_linewidth(2.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'baseline_comparison.png'), dpi=150)
    plt.close()

def plot_per_learner_type(pt, out):
    if not pt: return
    types = list(pt.keys())
    tc = ['#3b82f6','#8b5cf6','#ec4899','#f59e0b']
    sn = [t.replace('_','\n') for t in types]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('ATLAS Performance by Learner Archetype', fontsize=15, fontweight='bold', y=1.02)
    for ax, key, title, yl in [
        (axes[0], 'mean_knowledge', 'Knowledge Level', 'Knowledge [0-1]'),
        (axes[1], 'mastery_rate', 'Mastery Rate', 'Rate'),
        (axes[2], 'mean_reward', 'Mean Reward', 'Reward'),
    ]:
        vals = [pt[t][key] for t in types]
        bars = ax.bar(sn, vals, color=tc, edgecolor='white', linewidth=1.5)
        ax.set_title(title, fontweight='bold'); ax.set_ylabel(yl)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'per_learner_type.png'), dpi=150)
    plt.close()

def plot_trajectories(rd, out):
    data = load_json(rd, 'trajectories.json')
    if not data: return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Sample Knowledge Trajectories (Last 10 Episodes)', fontsize=14, fontweight='bold')
    cm = plt.cm.viridis
    for i, traj in enumerate(data):
        ax.plot(traj, color=cm(i/max(len(data)-1,1)), alpha=0.7, linewidth=1.5, label=f'Ep {i+1}')
    ax.axhline(y=0.8, color=C['accent'], linestyle='--', alpha=0.7, label='Mastery')
    ax.set_xlabel('Step'); ax.set_ylabel('Mean Knowledge'); ax.set_ylim(0,1.05)
    ax.legend(ncol=3, loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'knowledge_trajectories.png'), dpi=150)
    plt.close()

def plot_architecture(out):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.5); ax.axis('off')
    fig.patch.set_facecolor('white')
    def box(x, y, w, h, label, sub='', fc='#dbeafe', ec='#3b82f6', fontsize=11):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
            facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(rect)
        if sub:
            ax.text(x+w/2, y+h/2+0.15, label, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color='#1e293b')
            ax.text(x+w/2, y+h/2-0.2, sub, ha='center', va='center',
                    fontsize=8.5, color='#64748b', style='italic')
        else:
            ax.text(x+w/2, y+h/2, label, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color='#1e293b')
    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', color='#475569', lw=1.8))
    def label(x, y, txt):
        ax.text(x, y, txt, ha='center', va='center', fontsize=8, color='#64748b')
    box(3.5, 5.2, 4, 0.85, 'ATLAS Orchestrator', 'Multi-Agent Coordination Pipeline', '#e0e7ff', '#4f46e5', 12)
    bw, bh = 2.15, 0.85
    agents = [
        (0.35, 3.4, 'Curriculum Agent', 'LinUCB Contextual Bandit', '#dcfce7', '#16a34a'),
        (2.75, 3.4, 'Pedagogy Agent', 'PPO Policy Gradient', '#dbeafe', '#2563eb'),
        (5.15, 3.4, 'Difficulty Agent', 'Adaptive Calibration', '#fef3c7', '#d97706'),
        (7.55, 3.4, 'Assessment Agent', 'Bayesian Estimation', '#fce7f3', '#db2777'),
    ]
    for x, y, lbl, sub, fc, ec in agents:
        box(x, y, bw, bh, lbl, sub, fc, ec, 10)
    arrow(4.2, 5.2, 1.42, 4.25); arrow(4.8, 5.2, 3.82, 4.25)
    arrow(6.2, 5.2, 6.22, 4.25); arrow(6.8, 5.2, 8.62, 4.25)
    label(2.4, 4.85, 'topic?'); label(4.1, 4.85, 'strategy?')
    label(6.4, 4.85, 'adjust'); label(8.1, 4.85, 'assess')
    box(2.5, 1.7, 6, 0.85, 'Tutorial Environment', 'Simulated Learners + Curriculum Graph', '#f1f5f9', '#64748b', 11)
    arrow(3.82, 3.4, 4.2, 2.55); label(3.7, 2.95, 'action')
    arrow(6.8, 2.55, 6.8, 3.4); label(7.35, 2.95, 'reward\n+ state')
    tw = 1.25
    for x, y, lbl in [(2.6,0.4,'Fast\nVisual'),(4.1,0.4,'Slow\nMethodical'),(5.6,0.4,'Social\nLearner'),(7.1,0.4,'Practice\nOriented')]:
        box(x, y, tw, 0.8, lbl, '', '#fef2f2', '#ef4444', 9)
    arrow(5.5, 1.7, 5.5, 1.2)
    ax.text(5.5, 0.15, 'Learner Archetypes', ha='center', fontsize=9, color='#94a3b8')
    ax.set_title('ATLAS System Architecture', fontsize=16, fontweight='bold', pad=15, color='#1e293b')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'architecture.png'), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_summary_dashboard(m, comp, out):
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('ATLAS — Experimental Results Dashboard', fontsize=16, fontweight='bold', y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    # Knowledge
    ax = fig.add_subplot(gs[0,0])
    ax.plot(m['knowledge']['moving_avg'], color=C['knowledge'], lw=2)
    ax.axhline(y=0.8, color=C['accent'], ls='--', alpha=0.5)
    ax.set_title('Knowledge', fontweight='bold'); ax.set_ylabel('Mean Knowledge'); ax.set_ylim(0,1.05)
    # Mastery
    ax = fig.add_subplot(gs[0,1])
    ma = m['mastery_rate']['moving_avg']
    ax.fill_between(range(len(ma)),0,ma,alpha=0.3,color=C['mastery'])
    ax.plot(ma, color=C['mastery'], lw=2)
    ax.set_title('Mastery Rate', fontweight='bold'); ax.set_ylabel('Rate'); ax.set_ylim(0,1.05)
    # Reward
    ax = fig.add_subplot(gs[0,2])
    ax.plot(m['reward']['moving_avg'], color=C['atlas'], lw=2)
    ax.set_title('Reward', fontweight='bold'); ax.set_ylabel('Episode Reward')
    # Bars
    mk = ['Random','Fixed','ATLAS','Oracle']
    cl = [C['random'],C['fixed'],C['atlas'],C['oracle']]
    ax = fig.add_subplot(gs[1,0])
    v = [comp[k]['mean_knowledge'] for k in mk]
    bs = ax.bar(mk,v,color=cl); ax.set_title('Knowledge vs Baselines', fontweight='bold')
    for b,val in zip(bs,v): ax.text(b.get_x()+b.get_width()/2,val+0.01,f'{val:.2f}',ha='center',fontsize=8)
    ax = fig.add_subplot(gs[1,1])
    v = [comp[k]['mastery_rate'] for k in mk]
    bs = ax.bar(mk,v,color=cl); ax.set_title('Mastery vs Baselines', fontweight='bold')
    for b,val in zip(bs,v): ax.text(b.get_x()+b.get_width()/2,val+0.01,f'{val:.1%}',ha='center',fontsize=8)
    # Engagement
    ax = fig.add_subplot(gs[1,2])
    ma = m['engagement']['moving_avg']
    ax.plot(ma, color=C['engagement'], lw=2)
    ax.fill_between(range(len(ma)),0,ma,alpha=0.15,color=C['engagement'])
    ax.set_title('Engagement', fontweight='bold'); ax.set_ylabel('[0-1]'); ax.set_ylim(0,1.05)
    plt.savefig(os.path.join(out, 'summary_dashboard.png'), dpi=150)
    plt.close()

def main():
    rd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    out = os.path.join(rd, 'figures')
    os.makedirs(out, exist_ok=True)
    print("\n  Generating publication-quality figures...")
    r = load_results(rd)
    m = r['metrics']; comp = r['comparison']; pt = r['per_learner_type']
    plot_learning_curves(m, out); print("  [1/7] learning_curves.png")
    plot_baseline_comparison(comp, out); print("  [2/7] baseline_comparison.png")
    plot_per_learner_type(pt, out); print("  [3/7] per_learner_type.png")
    plot_trajectories(rd, out); print("  [4/7] knowledge_trajectories.png")
    plot_architecture(out); print("  [5/7] architecture.png")
    plot_summary_dashboard(m, comp, out); print("  [6/7] summary_dashboard.png")
    # engagement standalone
    fig, ax = plt.subplots(figsize=(10,4))
    ax.set_title('Learner Engagement Over Training', fontsize=14, fontweight='bold')
    ma = m['engagement']['moving_avg']
    ax.plot(ma, color=C['engagement'], lw=2)
    ax.fill_between(range(len(ma)),0,ma,alpha=0.15,color=C['engagement'])
    ax.set_xlabel('Episode'); ax.set_ylabel('Engagement [0-1]'); ax.set_ylim(0,1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'engagement.png'), dpi=150); plt.close()
    print("  [7/7] engagement.png")
    print("\n  All 7 figures generated!")

if __name__ == '__main__':
    main()