import re
from pathlib import Path

def main():
    filepath = Path("ranger_mini_v3.xml")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Change suspension joint to slide for proper vertical dynamics
    # Original (after my replacement): type="hinge" axis="0 1 0" stiffness="10500" damping="500"
    content = content.replace(
        'type="hinge" axis="0 1 0" stiffness="10500" damping="500"',
        'type="slide" axis="0 0 1" stiffness="10500" damping="500"'
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Fixed suspension joint type to slide.")

if __name__ == "__main__":
    main()
