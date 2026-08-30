"""
check_camera.py  —  Grab a frame from the front camera and save it as a PNG.
Run:  python check_camera.py
"""

import mujoco
import numpy as np

try:
    from PIL import Image
    USE_PIL = True
except ImportError:
    USE_PIL = False

import os

def main():
    model    = mujoco.MjModel.from_xml_path("scout_mini.xml")
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Step a bit so the robot settles on the ground
    for _ in range(100):
        mujoco.mj_step(model, data)

    renderer.update_scene(data, camera="front_cam")
    rgb = renderer.render()   # (480, 640, 3)  uint8

    out = "front_cam_frame.png"
    if USE_PIL:
        Image.fromarray(rgb).save(out)
        print(f"Saved {out}  shape={rgb.shape}  max_pixel={rgb.max()}")
    else:
        # Fallback: save raw bytes
        with open("front_cam_frame.raw", "wb") as f:
            f.write(rgb.tobytes())
        print(f"PIL not available — raw bytes saved.  shape={rgb.shape}")
        print("Install Pillow:  pip install Pillow")

    # Also test depth
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="front_cam")
    depth = renderer.render()  # (480, 640)  float32
    print(f"Depth  shape={depth.shape}  min={depth.min():.2f}m  max={depth.max():.2f}m")

if __name__ == "__main__":
    main()
