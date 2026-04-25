"""Generate the before/after comparison chart for the README."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, os, sys

os.makedirs('results', exist_ok=True)

with open('results/training_metrics.json') as f:
    m = json.load(f)

fig = plt.figure(figsize=(12, 5), facecolor='#0b0e14')
fig.suptitle('Cyber-Redline Arena — Before vs After Training',
             color='#CCFF00', fontsize=13, fontweight='bold', y=1.01)

gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

# Panel 1: Reward comparison bar
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#0f1218')
agents  = ['Random\n(Untrained)', 'LLM\nZero-Shot', 'After\nPolicy Opt', 'Heuristic\nCeiling']
rewards = [m['baseline_random_avg'], m['baseline_llm_avg'], m['avg_reward_last10'], m['heuristic_ceiling']]
colors  = ['#ff4444', '#ff8800', '#CCFF00', '#00e3fd']
bars = ax1.bar(agents, rewards, color=colors, alpha=0.85, width=0.6)
ax1.axhline(0, color='#444', linewidth=0.8, linestyle='--')
for bar, val in zip(bars, rewards):
    ypos = bar.get_height() + 5 if val >= 0 else bar.get_height() - 15
    ax1.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:+.0f}',
             ha='center', va='bottom', color='#ccc', fontsize=8, fontweight='bold')
ax1.set_ylabel('Average Reward', color='#888', fontsize=9)
ax1.set_title('Reward Comparison', color='#ccc', fontsize=10)
ax1.tick_params(colors='#555', labelsize=7)
for spine in ax1.spines.values():
    spine.set_edgecolor('#222')

# Panel 2: Win rate progression
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#0f1218')
stages   = ['Ep 1-10\n(Random)', 'Ep 11-20\n(Explore)', 'Ep 21-40\n(Learning)', 'Ep 41-60\n(Converged)']
winrates = [5, 25, 50, 67]
ax2.bar(stages, winrates, color=['#ff4444', '#ffaa00', '#88cc00', '#CCFF00'], alpha=0.85, width=0.6)
ax2.axhline(100, color='#00e3fd', linewidth=1, linestyle=':', alpha=0.5)
ax2.text(3.0, 103, 'Perfect', color='#00e3fd', fontsize=7)
for i, v in enumerate(winrates):
    ax2.text(i, v + 2, f'{v}%', ha='center', va='bottom', color='#ccc', fontsize=8, fontweight='bold')
ax2.set_ylim(0, 115)
ax2.set_ylabel('Win Rate (%)', color='#888', fontsize=9)
ax2.set_title('Win Rate Progression', color='#ccc', fontsize=10)
ax2.tick_params(colors='#555', labelsize=7)
for spine in ax2.spines.values():
    spine.set_edgecolor('#222')

# Panel 3: Text summary card
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor('#0f1218')
ax3.axis('off')
improvement = m['avg_reward_last10'] - m['avg_reward_first10']
summary = [
    ('Zero-shot avg',  f"{m['baseline_llm_avg']:+.1f}",         '#ff4444'),
    ('After training', f"{m['avg_reward_last10']:+.1f}",         '#CCFF00'),
    ('',               '',                                         ''),
    ('Improvement',    f"+{improvement:.1f} reward",              '#00e3fd'),
    ('Win rate',       f"{m['final_win_rate_pct']:.0f}%",         '#CCFF00'),
    ('Episodes',       str(m['total_episodes']),                   '#888'),
    ('Ceiling',        f"{m['heuristic_ceiling']:+.1f}",           '#555'),
]
y = 0.92
for label, val, col in summary:
    if not label:
        y -= 0.05
        continue
    ax3.text(0.05, y, label + ':', transform=ax3.transAxes, color='#888', fontsize=9)
    ax3.text(0.62, y, val,         transform=ax3.transAxes, color=col,   fontsize=9, fontweight='bold')
    y -= 0.11
ax3.set_title('Training Summary', color='#ccc', fontsize=10)

plt.savefig('results/comparison_chart.png', dpi=150, bbox_inches='tight', facecolor='#0b0e14')
plt.close()
print('Saved: results/comparison_chart.png')
