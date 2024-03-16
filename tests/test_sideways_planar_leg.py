import unittest
import numpy as np

from robot_gait_vis.leg import Leg
from robot_gait_vis.sideways_planar_leg import SidewaysPlanarLeg


class TestSidewaysPlanarLeg(unittest.TestCase):
    """Class to test the SidewaysPlanarLeg class."""

    def setUp(self) -> None:
        # Create leg that is to be used for each test
        self.sideways_planar_leg = SidewaysPlanarLeg((1, 2))

    def test_init(self) -> None:
        self.assertEqual(self.sideways_planar_leg.len_segments, (1, 2))
        self.assertEqual(self.sideways_planar_leg.num_joints, 2)
        # Check if instance is a subclass of Leg class
        self.assertIsInstance(self.sideways_planar_leg, Leg)
        self.assertIsInstance(self.sideways_planar_leg, SidewaysPlanarLeg)

    def test_forward_kinematics(self) -> None:
        # Test joint angles both at 0
        np.testing.assert_almost_equal(
            self.sideways_planar_leg.forward_kinematics((0, 0)),
            ((0, 1, 0), (0, 3, 0)),
            10)

        # Test with first joint at 90 degrees
        np.testing.assert_almost_equal(
            self.sideways_planar_leg.forward_kinematics((np.pi/2, 0)),
            ((0, 0, -1), (0, 0, -3)),
            10)

        # Test with second joint at 90 degrees
        np.testing.assert_almost_equal(
            self.sideways_planar_leg.forward_kinematics((0, np.pi/2)),
            ((0, 1, 0), (0, 1, -2)),
            10)

        # Test with both joints at 45 degrees
        np.testing.assert_almost_equal(
            self.sideways_planar_leg.forward_kinematics((np.pi/4, np.pi/4)),
            ((0, np.sqrt(2)/2, -np.sqrt(2)/2),
             (0, np.sqrt(2)/2, -np.sqrt(2)/2-2)),
            10)

        # Test with both joints at -45 degrees
        np.testing.assert_almost_equal(
            self.sideways_planar_leg.forward_kinematics((-np.pi/4, -np.pi/4)),
            ((0, np.sqrt(2)/2, np.sqrt(2)/2),
             (0, np.sqrt(2)/2, np.sqrt(2)/2+2)),
            10)

    def test_inverse_kinematics(self) -> None:
        # Test joint angles both at 0
        np.testing.assert_almost_equal(
            self.sideways_planar_leg.inverse_kinematics((0, 3, 0)),
            [(0, 0)],
            10)

        # Test with first joint at 90 degrees
        np.testing.assert_almost_equal(
            self.sideways_planar_leg.inverse_kinematics((0, 0, -3)),
            [(np.pi/2, 0)],
            10)

        # Test with second joint at 90 degrees
        np.testing.assert_almost_equal(
            self.sideways_planar_leg.inverse_kinematics((0, 1, -2)),
            [(0, np.pi/2), (2*np.arctan(2), -np.pi/2)],
            10)

        # Test with both joints at 45 degrees
        np.testing.assert_almost_equal(
            self.sideways_planar_leg.inverse_kinematics(
                (0, np.sqrt(2)/2, -np.sqrt(2)/2-2)),
            [(np.pi/4, np.pi/4),
             (np.arctan(2*np.sqrt(2) + 1)
                + np.arctan(np.sqrt(2)/(1+np.sqrt(2))),
              -np.pi/4)],
            10)

        # Test with both joints at -45 degrees
        np.testing.assert_almost_equal(
            self.sideways_planar_leg.inverse_kinematics(
                (0, np.sqrt(2)/2, np.sqrt(2)/2+2)),
            [(-np.arctan(2*np.sqrt(2) + 1)
                - np.arctan(np.sqrt(2) / (1+np.sqrt(2))),
             np.pi/4),
             (-np.pi/4, -np.pi/4)],
            10)

        # Test with second joint at 180 degrees
        np.testing.assert_almost_equal(
            self.sideways_planar_leg.inverse_kinematics(
                (0, -np.sqrt(2)/2, np.sqrt(2)/2)),
            [(np.pi/4, np.pi)],
            10)

        with self.assertRaises(ValueError):
            # Test with a x-coordinate given (out of workspace)
            self.sideways_planar_leg.inverse_kinematics((1, 0, 0))
            # Test with a coordinate outside of workspace but only in y-z
            self.sideways_planar_leg.inverse_kinematics((0, 10, 10))
