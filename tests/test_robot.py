import unittest

from robot_gait_vis.robot import Robot
from robot_gait_vis.planar_leg import PlanarLeg


class TestRobot(unittest.TestCase):
    """Class to test the Robot class."""

    def setUp(self) -> None:
        # Make a robot with a PlanarLeg
        self.planar_leg = PlanarLeg((1, 2))
        self.robot = Robot((1, 1, 1), self.planar_leg, 2)

    def test_init(self) -> None:
        # Test that an instance of Robot class has been made
        self.assertIsInstance(self.robot, Robot)

        # Test that inputs were registered
        self.assertEqual(self.robot.body_dims, (1, 1, 1))
        self.assertEqual(self.robot.leg_type, self.planar_leg)
        self.assertEqual(self.robot.num_legs, 2)

        # Test that variables are correctly generated
        self.assertEqual(self.robot.leg_names, ['L0', 'R0'])
        self.assertEqual(self.robot.joint_angles, {'L0': [], 'R0': []})
        self.assertEqual(self.robot.legs_pos_local, {'L0': [], 'R0': []})
        self.assertEqual(self.robot.legs_pos_global, {'L0': [], 'R0': []})
        self.assertEqual(self.robot.body_pos, [[0, 0, 0]])
        self.assertEqual(self.robot.leg_down, {'L0': [], 'R0': []})
        self.assertEqual(self.robot.frames, 0)

        # Repeat tests with different leg_nums and body_dims
        robot1 = Robot([3, 3, 3], self.planar_leg, 4)

        # Test that an instance of Robot class has been made
        self.assertIsInstance(robot1, Robot)

        # Test that inputs were registered
        self.assertEqual(robot1.body_dims, [3, 3, 3])
        self.assertEqual(robot1.leg_type, self.planar_leg)
        self.assertEqual(robot1.num_legs, 4)

        # Test that variables are correctly generated
        self.assertEqual(robot1.leg_names, ['L0', 'L1', 'R0', 'R1'])
        self.assertEqual(robot1.joint_angles,
                         {'L0': [], 'L1': [], 'R0': [], 'R1': []})
        self.assertEqual(robot1.legs_pos_local,
                         {'L0': [], 'L1': [], 'R0': [], 'R1': []})
        self.assertEqual(robot1.legs_pos_global,
                         {'L0': [], 'L1': [], 'R0': [], 'R1': []})
        self.assertEqual(robot1.body_pos, [[0, 0, 0]])
        self.assertEqual(robot1.leg_down,
                         {'L0': [], 'L1': [], 'R0': [], 'R1': []})
        self.assertEqual(robot1.frames, 0)
