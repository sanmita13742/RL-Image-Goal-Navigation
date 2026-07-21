import re
from pathlib import Path

def main():
    filepath = Path("ranger_mini_v3.xml")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove the armature from the wheel joints
    content = content.replace(
        'limited="false" damping="0.5" armature="0.02"',
        'limited="false" damping="0.5"'
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Removed drive armature.")

if __name__ == "__main__":
    main()
