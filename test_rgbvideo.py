import numpy as np
import open3d as o3d
import glob, os

traj = "data/pretraining_rollouts/50000_transitions/rgbd_frames/trajectory_00001"
rgb_files  = sorted(glob.glob(f"{traj}/step_*_rgb.npy"))
depth_files = sorted(glob.glob(f"{traj}/step_*_depth.npy"))

W, H = 64, 64
fovy_rad = np.deg2rad(42.0)
fy = (H / 2.0) / np.tan(fovy_rad / 2.0)
intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fy, fy, W/2, H/2)

flip = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

vis = o3d.visualization.Visualizer()
vis.create_window("D435 Trajectory", width=800, height=600)

pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

for rgb_f, depth_f in zip(rgb_files, depth_files):
    rgb   = np.load(rgb_f).astype(np.uint8)
    depth = np.load(depth_f).astype(np.float32)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth),
        depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
    )
    new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    new_pcd.transform(flip)

    pcd.points = new_pcd.points
    pcd.colors = new_pcd.colors
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

vis.destroy_window()