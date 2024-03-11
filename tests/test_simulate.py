import unittest

from robot_gait_vis.robot import Robot
from robot_gait_vis.planar_leg import PlanarLeg
from robot_gait_vis.simulate import Simulate


class TestSimulate(unittest.TestCase):
    """Class to test the Simulate class."""

    def setUp(self) -> None:
        # Make a biped with PlanarLegs for testing
        planar_leg = PlanarLeg([1, 2])
        self.planar_robot = Robot([1, 1, 1], planar_leg, 2)
        self.sim = Simulate(self.planar_robot, 0.5)


    def test_init(self):
        self.assertEqual(self.sim.dt, 0.5)
        self.assertIsInstance(self.sim.robot, Robot)

    def test_move_robot(self):
        pass
