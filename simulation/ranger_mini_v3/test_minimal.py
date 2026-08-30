import mujoco
from pathlib import Path

def main():
    model = mujoco.MjModel.from_xml_path("ranger_mini_v3.xml")
    data = mujoco.MjData(model)
    
    print("Running 100 steps...")
    try:
        for i in range(100):
            mujoco.mj_step(model, data)
            # Print if any qpos is NaN
            if sum(data.qpos) != sum(data.qpos):
                print(f"NaN detected at step {i}!")
                break
        print("Success.")
    except Exception as e:
        print("Exception:", e)

if __name__ == "__main__":
    main()
