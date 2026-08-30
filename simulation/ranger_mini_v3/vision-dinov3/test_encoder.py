
"""
test_encoder.py

Simple smoke test for the MINav perception module.
"""

from encoder_v2 import DINOEncoder

# -------------------------------------------------------
# Replace these paths with your own test images
# -------------------------------------------------------
TEST_IMAGES = [
    "wall2.jpg",
    "office2.jpg",
]

def main():

    print("=" * 60)
    print("Initializing DINO Encoder...")
    encoder = DINOEncoder()
    print("Done.\n")

    for image_path in TEST_IMAGES:

        result = encoder.process_image(image_path)

        print("-" * 60)
        print(f"Image                : {result.image_path}")
        print(f"Visual Representation: {tuple(result.visual_representation.shape)}")
        print(f"SSD                  : {result.ssd:.4f}")
        print(f"Valid Goal           : {result.valid_goal}")

    print("\nSmoke test completed successfully.")

if __name__ == "__main__":
    main()
