from typing import Tuple, List
import numpy as np

from robot_gait_vis.leg import Leg


class PlanarLeg(Leg):
    """A class to represent a 2 DOF planar leg 
    with sequential joints around the (Y-Y) axes"""

    def __init__(self, len_segments: Tuple[float, float]) -> None:
        super().__init__(len_segments)

    @property
    def num_joints(self) -> int:
        return 2  # This leg has 2 DOF

    def forward_kinematics(
            self,
            joint_angles: Tuple[float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:

        # knee
        x_knee = self.len_segments[0] * np.cos(joint_angles[0])
        y_knee = 0
        z_knee = -1 * self.len_segments[0] * np.sin(joint_angles[0])

        # end effector
        x_end = x_knee + self.len_segments[1] * np.cos(np.sum(joint_angles))
        y_end = 0
        z_end = z_knee - self.len_segments[1] * np.sin(np.sum(joint_angles))

        return ((x_knee, y_knee, z_knee), (x_end, y_end, z_end))

    def inverse_kinematics(
            self,
            end_effector_pos: Tuple[float, float, float]
    ) -> List[Tuple[float, float]]:

        l1 = self.len_segments[0]
        l2 = self.len_segments[1]

        X = end_effector_pos[0]
        Y = end_effector_pos[1]
        Z = end_effector_pos[2]

        sol = []
        cos_theta_2 = ((X**2 + Z**2) - (l1**2 + l2**2)) / (2*l1*l2)

        # No solutions
        if abs(cos_theta_2) > 1 or Y != 0:
            raise ValueError(f"Position {end_effector_pos} is outside "
                             + "the workspace of the leg, no solutions")
        # Edge case of theta_2 being 0 or pi, one solution
        if abs(cos_theta_2) - 1 < 0.0001 and abs(cos_theta_2) - 1 > -0.0001:
            sol.append((np.arctan2(-Z * cos_theta_2,
                                   X * cos_theta_2),
                        np.pi * (1 - (1+cos_theta_2)/2)))
        else:
            for i in range(2):
                theta_2 = np.arccos(cos_theta_2) * (-1)**i
                theta_1 = np.arctan2(-Z, X) \
                    - np.arctan2(l2 * np.sin(theta_2),
                                 l1 + l2 * np.cos(theta_2))
                sol.append((theta_1, theta_2))

        return sol
