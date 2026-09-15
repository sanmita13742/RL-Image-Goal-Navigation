"""
shared/ — MuJoCo-independent modules shared between simulation and real-world pipelines.

This package extracts portable code (pink noise, action normalization, drive commands,
exploration policies) so that both the MuJoCo simulation pipeline and the real-world
ROS 2 pipeline can reuse them without circular imports or MuJoCo dependencies.
"""

from shared.drive_command import DriveCommand
from shared.pink_noise import FFTColoredNoise, UniformColoredNoise
from shared.action_normalizer import ActionNormalizer
