import numpy as np
import open3d as o3d

# --- Load a frame ---
traj = "/home/dtc/dinesh_varun/16745/OCRL_Random_Agent/data/pretraining_rollouts/25000_transitions/2026-04-08_13-45-36/rgbd_frames/trajectory_00001"
rgb   = np.load(f"{traj}/step_000093_rgb.npy")    # (64, 64, 3) uint8
depth = np.load(f"{traj}/step_000093_depth.npy")  # (64, 64) float32, meters

# --- Camera intrinsics ---
W, H = 256, 256
fovy_rad = np.deg2rad(42.0)
fy = (H / 2.0) / np.tan(fovy_rad / 2.0)
fx = fy
cx, cy = W / 2.0, H / 2.0

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=W, height=H,
    fx=fx, fy=fy,
    cx=cx, cy=cy
)

# --- Build Open3D RGBD image ---
rgb_o3d   = o3d.geometry.Image(rgb.astype(np.uint8))
depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    rgb_o3d,
    depth_o3d,
    depth_scale=1.0,        # already in meters
    depth_trunc=5.0,        # clip depth beyond 5m
    convert_rgb_to_intensity=False
)

# --- Create point cloud ---
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

# Flip to standard orientation (MuJoCo uses Y-down)
pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

# --- Visualize ---
o3d.visualization.draw_geometries([pcd],
    window_name="D435 Point Cloud",
    width=800, height=600
)