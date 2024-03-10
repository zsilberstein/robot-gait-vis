from typing import List, Dict
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

    def test_create_ellipse_trajectory(self) -> None:
        # Get initial joint angles
        stance = 0.5
        height = 0.5
        start_thetas = {}
        gait_height = self.planar_robot.leg_type.forward_kinematics(
            np.deg2rad([45, 45]))[-1]
        for leg_name in self.planar_robot.leg_names:
            start_thetas[leg_name] = self.planar_robot.leg_type.inverse_kinematics(
                [stance, 0, gait_height[2]])[0]

        forward_start = self.planar_leg.forward_kinematics(
            start_thetas['L0'])[-1]

        # Create ellipse trajectory with 4 points and 0.5 duty factor
        stance_vec = [-stance, 0, 0]
        height_vec = [stance/2, 0, height]
        mid_thetas = self.planar_leg.inverse_kinematics(
            (forward_start[0]-stance, forward_start[1], forward_start[2]))[0]
        ideal = {'L0':
                 {'stance': [start_thetas['L0'], mid_thetas],
                  'swing': [mid_thetas, start_thetas['L0']]},
                 'R0':
                 {'stance': [start_thetas['R0'], mid_thetas],
                  'swing': [mid_thetas, start_thetas['R0']]}}
        actual = gait_gen.create_ellipse_trajectory(self.planar_robot,
                                                    start_thetas,
                                                    stance_vec,
                                                    height_vec,
                                                    stance_vec,
                                                    height_vec,
                                                    2,
                                                    0.5)
        self.compare_dict_of_dict_of_list(ideal, actual)

    def test_get_gait(self) -> None:
        # Get initial joint angles
        stance = 0.5
        height = 0.5
        start_thetas = {}
        gait_height = self.planar_robot.leg_type.forward_kinematics(
            np.deg2rad([45, 45]))[-1]
        for leg_name in self.planar_robot.leg_names:
            start_thetas[leg_name] = self.planar_robot.leg_type.inverse_kinematics(
                [stance, 0, gait_height[2]])[0]

        forward_start = self.planar_leg.forward_kinematics(
            start_thetas['L0'])[-1]

        # Create ellipse trajectory with 4 points and 0.5 duty factor
        stance_vec = [-stance, 0, 0]
        height_vec = [stance/2, 0, height]
        mid_thetas = self.planar_leg.inverse_kinematics(
            (forward_start[0]-stance, forward_start[1], forward_start[2]))[0]
        paths = gait_gen.create_ellipse_trajectory(self.planar_robot,
                                                   start_thetas,
                                                   stance_vec,
                                                   height_vec,
                                                   stance_vec,
                                                   height_vec,
                                                   2,
                                                   0.5)
        stance_start = {'L0': 0, 'R0': 0.5}

        ideal = [{'L0': mid_thetas,
                  'R0': start_thetas['L0']},
                 {'L0': start_thetas['R0'],
                  'R0': mid_thetas}]
        actual = gait_gen.get_gait(paths, stance_start)
        self.compare_list_of_dict_of_list(ideal, actual)

    def compare_dict_of_list(self,
                             d1: Dict[str, List[float]],
                             d2: Dict[str, List[float]],
                             decimal=10) -> None:
        """Compares two dictionaries with lists as the value for each item."""
        self.assertEqual(len(d1), len(d2),
                         f'List 1 has {len(d1)} keys but List 2 has {len(d2)} keys')
        for key in d1:
            self.assertTrue(key in d2, f'Key {key} is not in both dicts')
            np.testing.assert_almost_equal(d1[key], d2[key], decimal,
                                           f'Lists with key {key} are not almost equal')

    def compare_dict_of_dict_of_list(self,
                                     d1: Dict[str, Dict[str, List[float]]],
                                     d2: Dict[str, Dict[str, List[float]]],
                                     decimal=10) -> None:
        """Compares two dictionaries with dictionaries as the values for each item. 
        Each inner dictionary has lists for each item."""
        self.assertEqual(len(d1), len(d2),
                         f'List 1 has {len(d1)} keys but List 2 has {len(d2)} keys')
        for key in d1:
            self.assertTrue(key in d2, f'Key {key} is not in both dicts')
            self.compare_dict_of_list(d1[key], d2[key], decimal=decimal)

    def compare_list_of_dict_of_list(self,
                                     l1: List[Dict[str, List[float]]],
                                     l2: List[Dict[str, List[float]]],
                                     decimal=10) -> None:
        """Compares two lists with dictionaries at each index. 
        Each dictionary has lists for each item."""
        self.assertEqual(len(l1), len(l2))
        for d1, d2 in zip(l1, l2):
            self.compare_dict_of_list(d1, d2, decimal=decimal)
