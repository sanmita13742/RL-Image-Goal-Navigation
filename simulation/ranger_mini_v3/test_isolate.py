import mujoco
from pathlib import Path

def test_config(xml_content):
    try:
        model = mujoco.MjModel.from_xml_string(xml_content)
        data = mujoco.MjData(model)
        for i in range(100):
            mujoco.mj_step(model, data)
        return True
    except Exception as e:
        return False

def main():
    filepath = Path("ranger_mini_v3.xml")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Try reverting contact
    test_content = content.replace(
        '<geom friction="1.2 0.005 0.0001" condim="4" solimp="0.95 0.99 0.001" solref="0.01 1"/>',
        '<geom friction="1.2 0.005 0.0001" condim="4" solimp="0.9 0.95 0.001" solref="0.02 1"/>'
    )
    print("Reverted contact: ", test_config(test_content))
    
    # Try reverting suspension stiffness/damping
    test_content2 = content.replace(
        'stiffness="10500" damping="500"',
        'stiffness="1000" damping="10"'
    )
    print("Reverted suspension: ", test_config(test_content2))
    
    # Try increasing damping on steering
    test_content3 = content.replace(
        'damping="5.0" armature="1.0"',
        'damping="50.0" armature="1.0"'
    )
    print("Increased steering damping: ", test_config(test_content3))

if __name__ == "__main__":
    main()
