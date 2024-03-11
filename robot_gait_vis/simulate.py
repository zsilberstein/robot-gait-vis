from typing import List, Dict

from robot_gait_vis.robot import Robot


class Simulate():
    """Class to simulate the movement of a Robot."""

    def __init__(self, robot: Robot, dt: float) -> None:
        """
        Args:
            robot (Robot): Robot to simulate.
            dt (float): Time between each robot command in seconds.
        """
        self.robot = robot
        self.dt = dt

    def move_robot(self, thetas: Dict[str, List[float]]) -> None:
        """Updates the position of the robot given joint angles for the legs.

        Args:
            thetas (Dict[str, List[float]]): Dictionary of joint angles for 
            each leg to move to in radians.
        """
        return

    def animate(self, save_file: str):
        """Animates the history of the robot.

        Args:
            save_file (str): A .gif or a .mp4 file to save animation to.
        """
        return

    def plot_xyz_history(self) -> None:
        """Plots the position of the robot center of 
            body and the end-effector of each leg over time.
        """
        return

    def plot_joint_history(self) -> None:
        """Plots the joint angles of each leg over time.
        """
        return
