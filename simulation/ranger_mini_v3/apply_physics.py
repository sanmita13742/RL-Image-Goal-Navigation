import re
from pathlib import Path

def main():
    filepath = Path("ranger_mini_v3.xml")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Contact defaults
    content = content.replace(
        '<geom friction="1.2 0.005 0.0001" condim="4" solimp="0.9 0.95 0.001" solref="0.02 1"/>',
        '<geom friction="1.2 0.005 0.0001" condim="4" solimp="0.95 0.99 0.001" solref="0.01 1"/>'
    )

    # 2. Chassis Mass & Inertia
    content = content.replace(
        '<inertial pos="-0.02 -0.003 0.05" mass="75"\n                fullinertia="1.5 2.5 3.0 0.01 -0.02 -0.001"/>',
        '<inertial pos="-0.02 -0.003 0.05" mass="43"\n                diaginertia="1.3 1.3 1.3"/>'
    )

    # 3. Suspension joints
    content = content.replace(
        'type="hinge" axis="0 1 0" stiffness="1000" damping="10"',
        'type="hinge" axis="0 1 0" stiffness="10500" damping="500"'
    )

    # 4. Steering joints
    content = content.replace(
        'axis="0 0 -1"\n                 range="-3.14159 3.14159"\n                 damping="5.0" armature="0.05"',
        'axis="0 0 -1"\n                 limited="false"\n                 damping="5.0" armature="1.0"'
    )

    # 5. Wheel mass
    content = content.replace(
        'mass="8"\n                      fullinertia="0.04 0.07 0.04 0 0 0"',
        'mass="5"\n                      diaginertia="0.04 0.07 0.04"'
    )

    # 6. Wheel joints (armature)
    content = content.replace(
        'limited="false" damping="0.5"',
        'limited="false" damping="0.5" armature="0.02"'
    )

    # 7. Actuators
    content = re.sub(
        r'<position name="act_([fr][lr])_steer" joint="\1_steering_joint" kp="200" ctrlrange="-3.14159 3.14159"/>',
        r'<position name="act_\1_steer" joint="\1_steering_joint" kp="50" forcerange="-20 20" ctrlrange="-3.14159 3.14159"/>',
        content
    )
    content = re.sub(
        r'<velocity name="act_([fr][lr])_drive" joint="\1_wheel" kv="10"/>',
        r'<velocity name="act_\1_drive" joint="\1_wheel" kv="50" forcerange="-20 20" ctrlrange="-20 20"/>',
        content
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Updated ranger_mini_v3.xml successfully.")

if __name__ == "__main__":
    main()
