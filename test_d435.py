#!/usr/bin/env python3
"""Quick sanity-check for the D435 egocentric camera in the G1 MuJoCo model.

Usage:
    python test_d435.py                     # list cameras, render d435
    python test_d435.py --cam tracking      # render a different camera
    python test_d435.py --size 128          # render at 128x128
"""
import argparse
import os
import sys
import numpy as np

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco Python package not found. Install with: pip install mujoco")
    sys.exit(1)

XML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data/baseline_policies/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml",
)


def list_cameras(model):
    print(f"\nAvailable cameras ({model.ncam}):")
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        cam = model.cam(i)
        print(f"  [{i}] name={name!r}  fovy={cam.fovy[0]:.1f}°  mode={cam.mode[0]}")
    print()


def render_camera(model, data, cam_name: str, size: int):
    renderer = mujoco.Renderer(model, width=size, height=size)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam_name)

    renderer.disable_depth_rendering()
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    depth = renderer.render().copy().astype(np.float32)
    renderer.close()
    return rgb, depth


def save_results(rgb, depth, cam_name: str):
    rgb_path   = f"test_d435_{cam_name}_rgb.npy"
    depth_path = f"test_d435_{cam_name}_depth.npy"
    np.save(rgb_path, rgb)
    np.save(depth_path, depth)
    print(f"Saved: {rgb_path}  shape={rgb.shape}  dtype={rgb.dtype}")
    print(f"Saved: {depth_path}  shape={depth.shape}  dtype={depth.dtype}")
    print(f"Depth range: {depth.min():.3f} – {depth.max():.3f} m")

    try:
        import cv2
        png_path = f"test_d435_{cam_name}_rgb.png"
        cv2.imwrite(png_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved PNG: {png_path}")
    except ImportError:
        print("cv2 not available — RGB saved as .npy only")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam",  default="d435", help="Camera name to render")
    parser.add_argument("--size", type=int, default=256, help="Render size (square)")
    args = parser.parse_args()

    print(f"Loading model from:\n  {XML_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        print(f"ERROR loading XML: {e}")
        sys.exit(1)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    list_cameras(model)

    # Verify the requested camera exists
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
    if cam_id == -1:
        print(f"ERROR: camera {args.cam!r} not found in model. See list above.")
        sys.exit(1)

    print(f"Rendering {args.size}×{args.size} from camera {args.cam!r} ...")
    rgb, depth = render_camera(model, data, args.cam, args.size)
    save_results(rgb, depth, args.cam)
    print("\nAll OK. Check the saved .npy / .png files.")


if __name__ == "__main__":
    main()
