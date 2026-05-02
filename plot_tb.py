import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

log_file = "logs/rsl_rl/g1_velocity/2026-04-13_00-53-31/events.out.tfevents.1776041619.ip-172-31-75-216.28202.0"

try:
    ea = EventAccumulator(log_file)
    ea.Reload()

    tags = ea.Tags()['scalars']
    print("Available scalar tags:", tags)

    target_tag = None
    # Common RSL-RL reward scalar names
    candidates = ['Train/mean_reward', 'Perf/total_reward', 'Episode_Reward', 'Episode_return', 'mean_reward']
    
    for c in candidates:
        for t in tags:
            if c.lower() in t.lower():
                target_tag = t
                break
        if target_tag:
            break
            
    if not target_tag:
        for tag in tags:
            if 'reward' in tag.lower():
                target_tag = tag
                break

    if target_tag:
        print(f"Plotting tag: {target_tag}")
        events = ea.Scalars(target_tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        
        plt.figure(figsize=(10, 6))
        
        # Plot with a smooth modern style
        plt.style.use('ggplot')
        plt.plot(steps, vals, color='#2c7bb6', linewidth=2.5, alpha=0.9)
        
        plt.xlabel('Training Steps', fontsize=13, fontweight='bold')
        plt.ylabel('Mean Episode Reward', fontsize=13, fontweight='bold')
        plt.title('Agent Training Progress', fontsize=16, pad=15, fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('episode_reward_plot.png', dpi=300, bbox_inches='tight')
        print("Plot successfully saved to episode_reward_plot.png")
    else:
        print("Could not find a matching reward tag in the logs.")

except Exception as e:
    print(f"Error processing logs: {e}")
