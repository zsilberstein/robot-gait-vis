from typing import Tuple, List
import numpy as np

from robot_gait_vis.leg import Leg


class ThreeDOFLeg(Leg):
    """A class to represent a 3 DOF leg 
    with sequential joints around the (Z-X-X) axes"""

    def __init__(self, len_segments: Tuple[float, float, float]) -> None:
        super().__init__(len_segments)

    @property
    def num_joints(self) -> int:
        return 3  # This leg has 3 DOF

    def forward_kinematics(
            self,
            joint_angles: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float],
               Tuple[float, float, float],
               Tuple[float, float, float]]:
        # knee
        x_hip = np.sin(joint_angles[0]) * self.len_segments[0]
        y_hip = np.cos(joint_angles[0]) * self.len_segments[0]
        z_hip = 0

        # knee
        x_knee = np.sin(joint_angles[0]) * (self.len_segments[0]
                                            + self.len_segments[1]
                                            * np.cos(joint_angles[1]))
        y_knee = np.cos(joint_angles[0]) * (self.len_segments[0]
                                            + self.len_segments[1]
                                            * np.cos(joint_angles[1]))
        z_knee = -self.len_segments[1] * np.sin(joint_angles[1])

        # end effector
        x_end = np.sin(joint_angles[0]) * (self.len_segments[0]
                                           + self.len_segments[1]
                                           * np.cos(joint_angles[1])
                                           + self.len_segments[2]
                                           * np.cos(joint_angles[1]
                                                    + joint_angles[2]))
        y_end = np.cos(joint_angles[0]) * (self.len_segments[0]
                                           + self.len_segments[1]
                                           * np.cos(joint_angles[1])
                                           + self.len_segments[2]
                                           * np.cos(joint_angles[1]
                                                    + joint_angles[2]))
        z_end = -(self.len_segments[1]
                  * np.sin(joint_angles[1])
                  + self.len_segments[2]
                  * np.sin(joint_angles[1] + joint_angles[2]))

        # First "joint" is at the leg attachment point
        return ((x_hip, y_hip, z_hip),
                (x_knee, y_knee, z_knee),
                (x_end, y_end, z_end))

    def inverse_kinematics(
            self,
            end_effector_pos: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float]]:

        l1 = self.len_segments[0]
        l2 = self.len_segments[1]
        l3 = self.len_segments[2]

        x = end_effector_pos[0]
        y = end_effector_pos[1]
        z = end_effector_pos[2]

        sol = []
        theta_1 = np.arctan2(x, y)

        cos_theta_3 = (((x-l1*np.sin(theta_1))**2
                       + (y-l1*np.cos(theta_1))**2
                       + z**2
                       - (l2**2 + l3**2))
                       / (2*l2*l3))

        # No solutions
        if abs(cos_theta_3) > 1:
            raise ValueError(f"Position {end_effector_pos} is outside "
                             + "the workspace of the leg, no solutions")
        # Edge case of theta_3 being 0 or pi, one solution
        if abs(cos_theta_3) - 1 < 0.0001 and abs(cos_theta_3) - 1 > -0.0001:
            sol.append((theta_1,
                        np.arctan2(-z * cos_theta_3,
                                   np.sqrt((x-l1*np.sin(theta_1))**2
                                           + (y-l1*np.cos(theta_1))**2)),
                        np.pi * (1 - (1+cos_theta_3)/2)))
        else:
            for i in range(2):
                theta_3 = np.arccos(cos_theta_3) * (-1)**i
                theta_2 = (np.arctan2(-z, np.sqrt((x-l1*np.sin(theta_1))**2
                                                  + (y-l1*np.cos(theta_1))**2))
                           - np.arctan2(l3 * np.sin(theta_3),
                                        l2 + l3 * np.cos(theta_3)))
                sol.append((theta_1, theta_2, theta_3))

        return sol
