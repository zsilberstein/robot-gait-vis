import unittest
import numpy as np

from robot_gait_vis.robot import Robot
from robot_gait_vis.planar_leg import PlanarLeg
from robot_gait_vis import gait_gen


class TestGaitGen(unittest.TestCase):
    """Class to test the gait_gen module."""

    def setUp(self) -> None:
        # PlanarLeg and Robot
        self.planar_leg = PlanarLeg([1, 2])
        self.planar_robot = Robot([1, 1, 1], self.planar_leg, 2)

    def test_create_line(self) -> None:
        # Start at [0,0,-2] and move to [1,0,-2] with one point (odd case)
        start_angles = self.planar_leg.inverse_kinematics((0, 0, -2))[0]
        end_angles = self.planar_leg.inverse_kinematics((1, 0, -2))[0]
        np.testing.assert_almost_equal(
            gait_gen.create_line(self.planar_leg, start_angles, (1, 0, 0), 1),
            [start_angles, end_angles],
            10)

        # Start at [0,0,-2] and move to [1,0,-2] with two points (even case)
        mid_angles = self.planar_leg.inverse_kinematics((0.5, 0, -2))[0]
        np.testing.assert_almost_equal(
            gait_gen.create_line(self.planar_leg, start_angles, (1, 0, 0), 2),
            [start_angles, mid_angles, end_angles],
            10)

        with self.assertRaises(ValueError):
            # Move out of the workspace in the x-z plane
            gait_gen.create_line(self.planar_leg, start_angles, (5, 0, 5), 2)
            # Move out of the workspace by trying to move in y
            gait_gen.create_line(self.planar_leg, start_angles, (0, 1, 0), 2)

    def test_create_semi_ellipse(self) -> None:
        # Start at [0,0,-2] and move to [1,0,-2]
        #   with point of max curvature at [0.5, 0, -1.5] one point (odd case)
        start_angles = self.planar_leg.inverse_kinematics((0, 0, -2))[0]
        end_angles = self.planar_leg.inverse_kinematics((1, 0, -2))[0]
        np.testing.assert_almost_equal(
            gait_gen.create_semi_ellipse(self.planar_leg,
                                         start_angles,
                                         (1, 0, 0),
                                         (0.5, 0, 0.5),
                                         1),
            [start_angles, end_angles],
            10)

        # Start at [0,0,-2] and move to [1,0,-2] with
        #   point of max curvature at [0.5, 0, -1.5] two points (even case)
        height_angles = self.planar_leg.inverse_kinematics((0.5, 0, -1.5))[0]
        np.testing.assert_almost_equal(
            gait_gen.create_semi_ellipse(self.planar_leg,
                                         start_angles,
                                         (1, 0, 0),
                                         (0.5, 0, 0.5),
                                         2),
            [start_angles, height_angles, end_angles],
            10)

        with self.assertRaises(ValueError):
            # Move the base of the ellipse to extend out of the workspace
            gait_gen.create_semi_ellipse(self.planar_leg,
                                         start_angles,
                                         (5, 0, 5),
                                         (0, 0, 0),
                                         1)
            # Move the point of max curvature of
            #   the ellipse out of the workspace
            gait_gen.create_semi_ellipse(self.planar_leg,
                                         start_angles,
                                         (0, 0, 0),
                                         (5, 0, 5),
                                         2)
