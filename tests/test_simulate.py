from typing import Tuple, List, Dict
import unittest
import numpy as np

from robot_gait_vis.robot import Robot
from robot_gait_vis.planar_leg import PlanarLeg
from robot_gait_vis.simulate import Simulate
from robot_gait_vis import gait_gen
from robot_gait_vis.gaits import AlternatingGait


class TestSimulate(unittest.TestCase):
    """Class to test the Simulate class."""

    def setUp(self) -> None:
        # Make a biped with PlanarLegs for testing
        planar_leg = PlanarLeg((1, 2))
        self.planar_robot = Robot((1, 1, 1), planar_leg, 2)
        self.gait_type = AlternatingGait(self.planar_robot, 2)

        # Create gait for robot
        stance = 0.5
        height = 0.5
        start_thetas = {}
        gait_height = self.planar_robot.leg_type.forward_kinematics(
            np.deg2rad([45, 45]))[-1]
        for leg_name in self.planar_robot.leg_names:
            start_thetas[leg_name] = self.planar_robot.leg_type.inverse_kinematics(
                [stance, 0, gait_height[2]])[0]
        # Create ellipse trajectory with 2 points and 0.5 duty factor
        stance_vec = (-stance, 0, 0)
        height_vec = (stance/2, 0, height)
        self.gait = gait_gen.create_gait(self.planar_robot,
                                         self.gait_type,
                                         start_thetas,
                                         stance_vec,
                                         height_vec,
                                         stance_vec,
                                         height_vec)
        # Make sim
        self.sim = Simulate(self.planar_robot, 0.5, self.gait[0])

    def test_init(self):
        # Test that an instance of the Simulate class has been made
        self.assertIsInstance(self.sim, Simulate)
        self.assertEqual(self.sim.dt, 0.5)
        self.assertIsInstance(self.sim.robot, Robot)

        # Test joint angles
        joints = {'L0': [self.gait[0]['L0']], 'R0': [self.gait[0]['R0']]}
        self.compare_dict_list_list(self.sim.robot.joint_angles, joints)

        # Test local positions
        local = {'L0':
                 [self.planar_robot.leg_type.forward_kinematics(
                     self.gait[0]['L0'])],
                 'R0':
                 [self.planar_robot.leg_type.forward_kinematics(
                     self.gait[0]['R0'])]}
        self.compare_dict_list_list_tuple(self.sim.robot.legs_pos_local,
                                          local)

        # Test global positions
        global_leg = {'L0': [], 'R0': []}
        for leg, pos in global_leg.items():
            if leg[0] == 'R':
                y = -0.5
            else:
                y = 0.5
            for point in local[leg]:
                temp = list(point)
                for i, t in enumerate(temp):
                    inner = list(t)
                    inner[1] = y
                    temp[i] = inner
                pos.append(temp)
        self.compare_dict_list_list_tuple(
            self.sim.robot.legs_pos_global, global_leg)

        # Test leg attachment positions
        leg_attach = {'L0': [[0, 0.5, 0]],
                      'R0': [[0, -0.5, 0]]}
        self.assertEqual(self.sim.robot.leg_attach, leg_attach)

        # Test the body position
        body = [[0, 0, 0]]
        self.assertAlmostEqual(self.sim.robot.body_pos, body)

        # Test ground depth
        depth = self.planar_robot.leg_type.forward_kinematics(
            np.deg2rad([45, 45]))[-1][2]
        self.assertEqual(self.sim.ground, depth)

        # Test leg down
        down = {'L0': [True], 'R0': [True]}
        self.assertEqual(self.sim.robot.leg_down, down)

        self.assertEqual(self.sim.robot.frames, 1)

    def test_move_robot(self):
        # Move the robot
        for thetas in self.gait[1:]:
            self.sim.move_robot(thetas)

        # Test joint angles
        joints = {'L0': [self.gait[i]['L0'] for i in range(len(self.gait))],
                  'R0': [self.gait[i]['R0'] for i in range(len(self.gait))]}
        self.compare_dict_list_list(self.sim.robot.joint_angles, joints)

        # Test local positions
        local = {'L0':
                 [self.planar_robot.leg_type.forward_kinematics(self.gait[i]['L0'])
                  for i in range(len(self.gait))],
                 'R0':
                 [self.planar_robot.leg_type.forward_kinematics(self.gait[i]['R0'])
                  for i in range(len(self.gait))]}
        self.compare_dict_list_list_tuple(self.sim.robot.legs_pos_local, local)

        # Test global positions
        global_leg = {'L0': [], 'R0': []}
        for leg, pos in global_leg.items():
            if leg[0] == 'R':
                y = -0.5
            else:
                y = 0.5
            for point in local[leg]:
                temp = list(point)
                for i, t in enumerate(temp):
                    inner = list(t)
                    inner[1] = y
                    temp[i] = inner
                pos.append(temp)
        self.compare_dict_list_list_tuple(
            self.sim.robot.legs_pos_global, global_leg)

        # Test leg attachment positions
        leg_attach = {'L0': [[0, 0.5, 0], [0, 0.5, 0]],
                      'R0': [[0, -0.5, 0], [0, -0.5, 0]]}
        self.assertEqual(self.sim.robot.leg_attach, leg_attach)

        # Test the body position
        body = [[0, 0, 0], [0, 0, 0]]
        self.assertAlmostEqual(self.sim.robot.body_pos, body)

        # Test leg down
        down = {'L0': [True, True], 'R0': [True, True]}
        self.assertEqual(self.sim.robot.leg_down, down)

        self.assertEqual(self.sim.robot.frames, 2)

    def compare_dict_list_list(self,
                               d1: Dict[str, List[List[float]]],
                               d2: Dict[str, List[List[float]]],
                               decimal=10) -> None:
        self.assertEqual(len(d1), len(d2),
                         f'List 1 has {len(d1)} keys but List 2 has {len(d2)} keys')
        for key in d1:
            self.assertTrue(key in d2, f'Key {key} is not in both dicts')
            self.assertEqual(len(d1[key]), len(d2[key]))
            for i in range(len(d1[key])):
                self.assertEqual(len(d1[key][i]), len(d2[key][i]))
                np.testing.assert_almost_equal(d1[key][i], d2[key][i], decimal,
                                               f'Lists with key {key}, {i} are not almost equal')

    def compare_dict_list_list_tuple(self,
                                     d1: Dict[str, List[List[Tuple[float, ...]]]],
                                     d2: Dict[str, List[List[Tuple[float, ...]]]],
                                     decimal=10) -> None:
        self.assertEqual(len(d1), len(d2),
                         f'List 1 has {len(d1)} keys but List 2 has {len(d2)} keys')
        for key in d1:
            self.assertTrue(key in d2, f'Key {key} is not in both dicts')
            self.assertEqual(len(d1[key]), len(d2[key]))
            for i in range(len(d1[key])):
                self.assertEqual(len(d1[key][i]), len(d2[key][i]))
                for j in range(len(d1[key][i])):
                    self.assertEqual(len(d1[key][i][j]), len(d2[key][i][j]))
                    np.testing.assert_almost_equal(d1[key][i][j], d2[key][i][j],
                                                   decimal,
                                                   f'Tuples at index {key}, {i}, {j} are not almost equal')
