from abc import ABC, abstractmethod
from typing import Tuple, List


class Leg(ABC):
    """An abstract class to define key parameters of a robot leg."""

    def __init__(self, len_segments: Tuple[float, ...]) -> None:
        """
        Args:
            len_segments (Tuple[float, ...]): Length of sequential 
            leg segments in meters.
        """
        self.len_segments = len_segments

    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Property that stores the number of joints the leg has."""
        return

    @abstractmethod
    def forward_kinematics(
        self,
        joint_angles: Tuple[float, ...]
    ) -> Tuple[Tuple[float, float, float], ...]:
        """Computes the location of each joint and the end-effector.

        Args:
            joint_angles (Tuple[float, ...]): Current joint angles of the leg 
            in radians.

        Returns:
            Tuple[Tuple[float, float, float], ...]: A tuple of tuples representing 
            coordinates of each joint. First entry is the first joint and 
            last entry is for the end-effector. Each coordinate is 
            listed as (X, Y, Z) in meters.
        """
        return

    @abstractmethod
    def inverse_kinematics(
        self,
        end_effector_pos: Tuple[float, float, float]
    ) -> List[Tuple[float, ...]]:
        """Computes the joint angles of the leg given the 
        end-effector position.

        Args:
            end_effector_pos (Tuple[float, float, float]): 
            Coordinates of end-effector as (X, Y, Z) in meters.

        Raises:
            ValueError: Raises error if input end_effector_pos 
            is outside the workspace of the leg.

        Returns:
            List[Tuple[float, ...]]: A list of solutions of joint 
            angles of the leg in radians.
        """
        return
