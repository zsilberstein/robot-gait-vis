"""A python package to simulate and visualize different walking robot 
configurations and gaits.
"""
__all__ = ['Leg', 'PlanarLeg', 'ThreeDOFLeg', 'Gait', 'AlternatingGait',
           'RippleGait', 'WaveGait', 'Robot', 'Simulate',
           'create_line', 'create_semi_ellipse', 'create_ellipse_trajectory',
           'trajectory_to_gait', 'create_gait']
from robot_gait_vis.leg import Leg
from robot_gait_vis.planar_leg import PlanarLeg
from robot_gait_vis.three_dof_leg import ThreeDOFLeg
from robot_gait_vis.gaits import Gait, AlternatingGait, RippleGait, WaveGait
from robot_gait_vis.robot import Robot
from robot_gait_vis.simulate import Simulate
from robot_gait_vis.gait_gen import create_line, create_semi_ellipse, \
    create_ellipse_trajectory, trajectory_to_gait, create_gait
