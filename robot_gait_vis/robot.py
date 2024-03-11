from typing import Tuple

from robot_gait_vis.leg import Leg


class Robot:
    """A class to represent a robot"""

    def __init__(
            self,
            body_dims: Tuple[float, float, float],
            leg_type: Leg,
            num_legs: int
    ) -> None:
        """
        Args:
            body_dims (Tuple[float, float, float]): Dimensions of the 
            robot body specified as (X,Y,Z) in meters.
            leg_type (Leg): Type of leg of each leg of the robot.
            num_legs (int): Number of legs the robot has. Must be positive and even.

        Raises:
            ValueError: Raises error if num_legs is not positive or even.
        """
        self.body_dims = body_dims
        self.leg_type = leg_type
        if num_legs < 2 or num_legs % 2 != 0:
            raise ValueError("num_legs must be a multiple of 2 and positive")
        self.num_legs = num_legs

        # Build List of leg names
        self.leg_names = ['L' + str(i) for i in range(self.num_legs//2)] \
            + ['R' + str(i) for i in range(self.num_legs//2)]
        self.joint_angles = {leg_name: [] for leg_name in self.leg_names}
        self.legs_pos_local = {leg_name: [] for leg_name in self.leg_names}
        self.legs_pos_global = {leg_name: [] for leg_name in self.leg_names}
        self.body_pos = [[0, 0, 0]]
        self.leg_down = {leg_name: [] for leg_name in self.leg_names}
        self.frames = 0

        self.leg_attach = {}
        for leg_name in self.leg_names:
            self.leg_attach[leg_name] = []
            x = (-1/2 + (1/2 + int(leg_name[1:])) /
                 (self.num_legs / 2)) * self.body_dims[0]
            y = self.body_dims[1]/2
            if leg_name[0] == 'R':
                y *= -1
            self.leg_attach[leg_name].append([x, y, 0])
