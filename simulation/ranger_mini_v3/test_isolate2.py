import mujoco
from pathlib import Path
import sys
import os
import contextlib
import io

def test_config(xml_content):
    # Redirect stderr to capture mujoco warnings
    stderr_capture = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr_capture
    
    warned = False
    try:
        model = mujoco.MjModel.from_xml_string(xml_content)
        data = mujoco.MjData(model)
        for i in range(100):
            mujoco.mj_step(model, data)
    except Exception as e:
        warned = True
    
    sys.stderr = old_stderr
    output = stderr_capture.getvalue()
    if "WARNING" in output:
        warned = True
        
    return not warned # True if it passed without warning

def main():
    filepath = Path("ranger_mini_v3.xml")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Base (Current broken)
    print("Base XML (Current):", test_config(content))

    # 2. Revert contact
    test_content1 = content.replace(
        '<geom friction="1.2 0.005 0.0001" condim="4" solimp="0.95 0.99 0.001" solref="0.01 1"/>',
        '<geom friction="1.2 0.005 0.0001" condim="4" solimp="0.9 0.95 0.001" solref="0.02 1"/>'
    )
    print("Reverted contact:", test_config(test_content1))
    
    # 3. Revert suspension
    test_content2 = content.replace(
        'stiffness="10500" damping="500"',
        'stiffness="1000" damping="10"'
    )
    print("Reverted suspension:", test_config(test_content2))
    
    # 4. Revert Steering Actuator Armature
    test_content3 = content.replace(
        'armature="1.0"',
        'armature="0.05"'
    )
    print("Reverted steering armature:", test_config(test_content3))
    
    # 5. Revert Drive Actuator Armature
    test_content4 = content.replace(
        'armature="0.02"',
        ''
    )
    print("Reverted drive armature:", test_config(test_content4))

if __name__ == "__main__":
    main()
