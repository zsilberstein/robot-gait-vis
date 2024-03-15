from typing import Dict
import unittest

from robot_gait_vis.planar_leg import PlanarLeg
from robot_gait_vis.robot import Robot
from robot_gait_vis.gaits import Gait
from robot_gait_vis.gaits import AlternatingGait
from robot_gait_vis.gaits import RippleGait
from robot_gait_vis.gaits import WaveGait


class TestAlternatingGait(unittest.TestCase):
    """Class to test the AlternatingGait class."""

    def test_init(self) -> None:
        # PlanarLeg and Robots with different leg numbers
        planar_leg = PlanarLeg([1, 2])
        planar_robot_two = Robot([1, 1, 1], planar_leg, 2)
        planar_robot_six = Robot([1, 1, 1], planar_leg, 6)

        alternating_gait_two = AlternatingGait(planar_robot_two, 10)
        self.assertEqual(alternating_gait_two.robot, planar_robot_two)
        self.assertEqual(alternating_gait_two.points, 10)
        # Check if instance is a subclass of Gait class
        self.assertIsInstance(alternating_gait_two, Gait)
        self.assertIsInstance(alternating_gait_two, AlternatingGait)

        # Test duty factor
        self.assertEqual(alternating_gait_two.duty_factor, 1/2)
        # Test stance start
        stance_start_two = {'R0': 0, 'L0': 0.5}
        self.assertEqual(alternating_gait_two.stance_start, stance_start_two)

        alternating_gait_six = AlternatingGait(planar_robot_six, 10)
        self.assertEqual(alternating_gait_six.robot, planar_robot_six)
        self.assertEqual(alternating_gait_six.points, 10)
        # Check if instance is a subclass of Gait class
        self.assertIsInstance(alternating_gait_six, Gait)
        self.assertIsInstance(alternating_gait_six, AlternatingGait)

        # Test duty factor
        self.assertEqual(alternating_gait_six.duty_factor, 1/2)
        # Test stance start
        stance_start_six = {'R0': 0, 'R1': 0.5, 'R2': 0,
                            'L0': 0.5, 'L1': 0, 'L2': 0.5}
        self.assertEqual(alternating_gait_six.stance_start, stance_start_six)


class TestRippleGait(unittest.TestCase):
    """Class to test the RippleGait class."""

    def test_init(self) -> None:
        # PlanarLeg and Robots with different leg numbers
        planar_leg = PlanarLeg([1, 2])
        planar_robot_two = Robot([1, 1, 1], planar_leg, 2)
        planar_robot_six = Robot([1, 1, 1], planar_leg, 6)

        with self.assertRaises(ValueError):
            RippleGait(planar_robot_two, 10)

        ripple_gait_six = RippleGait(planar_robot_six, 10)
        self.assertEqual(ripple_gait_six.robot, planar_robot_six)
        self.assertEqual(ripple_gait_six.points, 10)
        # Check if instance is a subclass of Gait class
        self.assertIsInstance(ripple_gait_six, Gait)
        self.assertIsInstance(ripple_gait_six, RippleGait)

        # Test duty factor
        self.assertAlmostEqual(ripple_gait_six.duty_factor, 2/3)
        # Test stance start
        stance_start_six = {'R0': 0, 'R1': 1/3, 'R2': 2/3,
                            'L0': 1/3, 'L1': 2/3, 'L2': 0}
        self.compare_dict_of_floats(
            ripple_gait_six.stance_start, stance_start_six)

    def compare_dict_of_floats(self,
                               d1: Dict[str, float],
                               d2: Dict[str, float]):
        self.assertEqual(len(d1), len(d2))
        for key in d1:
            self.assertTrue(key in d2, f'Key {key} is not in both dicts')
            self.assertAlmostEqual(d1[key], d2[key])


class TestWaveGait(unittest.TestCase):
    """Class to test the WaveGait class."""

    def test_init(self) -> None:
        # PlanarLeg and Robots with different leg numbers
        planar_leg = PlanarLeg([1, 2])
        planar_robot_two = Robot([1, 1, 1], planar_leg, 2)
        planar_robot_six = Robot([1, 1, 1], planar_leg, 6)

        wave_gait_two = WaveGait(planar_robot_two, 10)
        self.assertEqual(wave_gait_two.robot, planar_robot_two)
        self.assertEqual(wave_gait_two.points, 10)
        # Check if instance is a subclass of Gait class
        self.assertIsInstance(wave_gait_two, Gait)
        self.assertIsInstance(wave_gait_two, WaveGait)

        # Test duty factor
        self.assertEqual(wave_gait_two.duty_factor, 1/2)
        # Test stance start
        stance_start_two = {'R0': 0, 'L0': 0.5}
        self.assertEqual(wave_gait_two.stance_start, stance_start_two)

        wave_gait_six = WaveGait(planar_robot_six, 10)
        self.assertEqual(wave_gait_six.robot, planar_robot_six)
        self.assertEqual(wave_gait_six.points, 10)
        # Check if instance is a subclass of Gait class
        self.assertIsInstance(wave_gait_six, Gait)
        self.assertIsInstance(wave_gait_six, WaveGait)

        # Test duty factor
        self.assertEqual(wave_gait_six.duty_factor, 5/6)
        # Test stance start
        stance_start_six = {'R0': 0, 'R1': 1/6, 'R2': 1/3,
                            'L0': 1/2, 'L1': 2/3, 'L2': 5/6}
        self.compare_dict_of_floats(wave_gait_six.stance_start,
                                    stance_start_six)
    
    def compare_dict_of_floats(self,
                               d1: Dict[str, float],
                               d2: Dict[str, float]):
        self.assertEqual(len(d1), len(d2))
        for key in d1:
            self.assertTrue(key in d2, f'Key {key} is not in both dicts')
            self.assertAlmostEqual(d1[key], d2[key])  
