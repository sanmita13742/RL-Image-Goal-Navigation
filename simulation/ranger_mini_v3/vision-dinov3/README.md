# MINav Vision Module

This package implements the **perception module** for the MINav project.

It provides a frozen **DINOv3** visual encoder together with the **Spatial Standard Deviation (SSD)** based goal filtering described in the MINav paper.

The module is designed to be imported by other components (Dataset Builder, RL, Evaluation) without requiring them to know any implementation details.

---

# Features

- Frozen pretrained DINOv3 encoder
- Automatic image preprocessing
- Visual representation extraction (φ(o))
- SSD-based goal filtering
- Batch processing support
- Simple Python package interface

---

# Installation

This vision module is currently maintained on the **`vision` branch**.

After cloning the repository, switch to the `vision` branch:

```bash
git clone https://github.com/sanmita13742/RL-Image-Goal-Navigation.git
cd RL-Image-Goal-Navigation
git checkout vision
```

Install the dependencies from the `requirements.txt` file located in the current directory:

```bash
pip install -r requirements.txt
```

---

# Usage

The encoder is provided directly by the vision module.

Import the encoder:

```python
from encoder_v2 import DINOEncoder
```

Create the encoder:

```python
encoder = DINOEncoder()
```

Process a single image:

```python
result = encoder.process_image("office.jpg")
```

Access the outputs:

```python
print(result.visual_representation.shape)
print(result.ssd)
print(result.valid_goal)
```

---

# Batch Processing

```python
from encoder_v2 import DINOEncoder

encoder = DINOEncoder()

images = [
    "frame1.jpg",
    "frame2.jpg",
    "frame3.jpg"
]

results = encoder.process_batch(images)

for result in results:
    print(result.ssd, result.valid_goal)
```

---

# Output

`process_image()` returns a `PerceptionResult` object.

```python
result.image_path
result.visual_representation
result.ssd
result.valid_goal
```

### Description

| Field                   | Description                                 |
| ----------------------- | ------------------------------------------- |
| `image_path`            | Input image path                            |
| `visual_representation` | DINOv3 visual feature φ(o)                  |
| `ssd`                   | Spatial Standard Deviation score            |
| `valid_goal`            | Whether the image passes SSD goal filtering |

---

# Pipeline

```text
Image
    │
Resize (448 × 784)
    │
Frozen DINOv3
    │
Visual Representation φ(o)
    │
Patch Embeddings
    │
14 × 25 Center Crop
    │
SSD Computation
    │
Goal Decision
```

---

# Model Configuration

| Parameter     | Value           |
| ------------- | --------------- |
| Model         | DINOv3 ViT-S/16 |
| Image Size    | 448 × 784       |
| Patch Grid    | 28 × 49         |
| Center Crop   | 14 × 25         |
| SSD Threshold | 0.02            |

---

# Notes

- The DINOv3 encoder is **frozen** during inference (`eval()` mode with gradients disabled).
- The SSD implementation follows the validated **Method 2** used during development.
- The module automatically performs image preprocessing and feature extraction.
- Consumers of this package should only use the public API (`DINOEncoder`) and should not rely on internal helper methods.

---

# Example

```python
from encoder_v2 import DINOEncoder

encoder = DINOEncoder()

result = encoder.process_image("office.jpg")

if result.valid_goal:
    feature = result.visual_representation
    print("Valid goal found!")
else:
    print("Image rejected.")
```

---

# For Developers

The recommended import is:

```python
from encoder_v2 import DINOEncoder
```

Avoid modifying or relying on internal helper methods unless you are modifying the perception module itself. This keeps the public interface stable even if the internal implementation changes.