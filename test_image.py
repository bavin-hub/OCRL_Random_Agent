import cv2
import numpy as np

array = np.load('/home/dtc/dinesh_varun/16745/OCRL_Random_Agent/data/pretraining_rollouts/25000_transitions/2026-04-08_13-45-36/rgbd_frames/trajectory_00001/step_000000_rgb.npy')

image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
import numpy as np
import matplotlib.pyplot as plt

traj = "data/pretraining_rollouts/50000_transitions/rgbd_frames/trajectory_00001"
for step in range(5):
    rgb   = np.load(f"{traj}/step_{step:06d}_rgb.npy")
    depth = np.load(f"{traj}/step_{step:06d}_depth.npy")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB  step={step}")
    axes[1].imshow(depth, cmap="plasma")
    axes[1].set_title(f"Depth step={step}  range [{depth.min():.2f}, {depth.max():.2f}]m")
    plt.tight_layout()
    plt.savefig(f"frame_{step:03d}.png", dpi=100)
    plt.close()
'''