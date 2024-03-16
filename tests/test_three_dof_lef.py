import unittest
import numpy as np

from robot_gait_vis.leg import Leg
from robot_gait_vis.three_dof_leg import ThreeDOFLeg


class TestThreeDOFLeg(unittest.TestCase):
    """Class to test the ThreeDOFLeg class."""

    def setUp(self) -> None:
        # Create leg that is to be used for each test
        self.three_dof_leg = ThreeDOFLeg((1, 1, 2))

    def test_init(self) -> None:
        self.assertEqual(self.three_dof_leg.len_segments, (1, 1, 2))
        self.assertEqual(self.three_dof_leg.num_joints, 3)
        # Check if instance is a subclass of Leg class
        self.assertIsInstance(self.three_dof_leg, Leg)
        self.assertIsInstance(self.three_dof_leg, ThreeDOFLeg)

    def test_forward_kinematics(self) -> None:
        # Test joint angles all at 0
        np.testing.assert_almost_equal(
            self.three_dof_leg.forward_kinematics((0, 0, 0)),
            ((0, 1, 0), (0, 2, 0), (0, 4, 0)),
            10)

        # Test with first joint at 90 degrees
        np.testing.assert_almost_equal(
            self.three_dof_leg.forward_kinematics((np.pi/2, 0, 0)),
            ((1, 0, 0), (2, 0, 0), (4, 0, 0)),
            10)

        # Test with second joint at 90 degrees
        np.testing.assert_almost_equal(
            self.three_dof_leg.forward_kinematics((0, np.pi/2, 0)),
            ((0, 1, 0), (0, 1, -1), (0, 1, -3)),
            10)

        # Test with third joint at 90 degrees
        np.testing.assert_almost_equal(
            self.three_dof_leg.forward_kinematics((0, 0, np.pi/2)),
            ((0, 1, 0), (0, 2, 0), (0, 2, -2)),
            10)

        # Test with all joints at 45 degrees
        np.testing.assert_almost_equal(
            self.three_dof_leg.forward_kinematics((np.pi/4, np.pi/4, np.pi/4)),
            ((np.sqrt(2)/2, np.sqrt(2)/2, 0),
             (1/2+np.sqrt(2)/2, 1/2+np.sqrt(2)/2, -np.sqrt(2)/2),
             (1/2+np.sqrt(2)/2, 1/2+np.sqrt(2)/2, -np.sqrt(2)/2-2)),
            10)

        # Test with all joints at -45 degrees
        np.testing.assert_almost_equal(
            self.three_dof_leg.forward_kinematics(
                (-np.pi/4, -np.pi/4, -np.pi/4)),
            ((-np.sqrt(2)/2, np.sqrt(2)/2, 0),
             (-1/2-np.sqrt(2)/2, 1/2+np.sqrt(2)/2, np.sqrt(2)/2),
             (-1/2-np.sqrt(2)/2, 1/2+np.sqrt(2)/2, np.sqrt(2)/2+2)),
            10)

    def test_inverse_kinematics(self) -> None:
        # Test joint angles all at 0
        np.testing.assert_almost_equal(
            self.three_dof_leg.inverse_kinematics((0, 4, 0)),
            [(0, 0, 0)],
            10)

        # Test with first joint at 90 degrees
        np.testing.assert_almost_equal(
            self.three_dof_leg.inverse_kinematics((4, 0, 0)),
            [(np.pi/2, 0, 0)],
            10)

        # Test with second joint at 90 degrees
        np.testing.assert_almost_equal(
            self.three_dof_leg.inverse_kinematics((0, 1, -3)),
            [(0, np.pi/2, 0)],
            10)

        # Test with third joint at 90 degrees
        np.testing.assert_almost_equal(
            self.three_dof_leg.inverse_kinematics((0, 2, -2)),
            [(0, 0, np.pi/2), (0, 2*np.arctan(2), -np.pi/2)],
            10)

        # Test with all joints at 45 degrees
        np.testing.assert_almost_equal(
            self.three_dof_leg.inverse_kinematics(
                (1/2+np.sqrt(2)/2, 1/2+np.sqrt(2)/2, -np.sqrt(2)/2-2)),
            [(np.pi/4, np.pi/4, np.pi/4),
             (np.pi/4, np.arctan(2*np.sqrt(2) + 1)
                + np.arctan(np.sqrt(2)/(1+np.sqrt(2))),
              -np.pi/4)],
            10)

        # Test with all joints at -45 degrees
        np.testing.assert_almost_equal(
            self.three_dof_leg.inverse_kinematics(
                (-1/2-np.sqrt(2)/2, 1/2+np.sqrt(2)/2, np.sqrt(2)/2+2)),
            [(-np.pi/4, -np.arctan(2*np.sqrt(2) + 1)
                - np.arctan(np.sqrt(2) / (1+np.sqrt(2))),
             np.pi/4),
             (-np.pi/4, -np.pi/4, -np.pi/4)],
            10)

        # Test with third joint at 180 degrees
        np.testing.assert_almost_equal(
            self.three_dof_leg.inverse_kinematics(
                ((np.sqrt(2)-1)/2, (np.sqrt(2)-1)/2, np.sqrt(2)/2)),
            [(np.pi/4, np.pi/4, np.pi)],
            10)

        with self.assertRaises(ValueError):
            # Test with a coordinate outside of workspace
            self.three_dof_leg.inverse_kinematics((10, 10, 10))
