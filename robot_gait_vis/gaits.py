from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import dataclass

from robot_gait_vis.robot import Robot


@dataclass
class Gait(ABC):
    """An abstract class to define key parameters of gait.

    Args:
        robot (Robot): Robot the gait is to be applied to.
        points (int): Total number of joint angles to generate for each leg 
        per gait cycle.
    """
    robot: Robot
    points: int

    @property
    @abstractmethod
    def duty_factor(self) -> float:
        """Fraction of the gait cycle each leg is on the ground."""
        return

    @property
    @abstractmethod
    def stance_start(self) -> Dict[str, float]:
        """Dictionary specifying the fraction of the gait cycle each leg is 
        to begin the stance phase.
        """
        return


@dataclass
class AlternatingGait(Gait):
    """A class to define an alternating gait.

    Args:
        robot (Robot): Robot the gait is to be applied to.
        points (int): Total number of joint angles to generate for each leg 
        per gait cycle.
    """

    def __post_init__(self):
        # Init stance start here so that it is only generated once
        self._stance = {}
        for leg_name in self.robot.leg_names:
            leg_num = int(leg_name[1:])
            if leg_name[0] == 'L':
                leg_num += 1  # Makes left and right alternate
            self._stance[leg_name] = (leg_num % 2) * self.duty_factor

    @property
    def duty_factor(self) -> float:
        return 1/2

    @property
    def stance_start(self) -> Dict[str, float]:
        return self._stance


@dataclass
class RippleGait(Gait):
    """A class to define a ripple gait.

    Args:
        robot (Robot): Robot the gait is to be applied to.
        points (int): Total number of joint angles to generate for each leg 
        per gait cycle.
    """

    def __post_init__(self):
        if self.robot.num_legs < 4:
            raise ValueError(
                'Ripple gait requires a Robot with at least 4 legs')
        # Init stance start here so that it is only generated once
        self._stance = {}
        for leg_name in self.robot.leg_names:
            leg_num = int(leg_name[1:])
            if leg_name[0] == 'L':
                leg_num += 1  # Makes left and right alternate
            self._stance[leg_name] = ((leg_num % (self.robot.num_legs/2))
                                      * (1 - self.duty_factor))

    @property
    def duty_factor(self) -> float:
        # 2 legs swing at a time
        return 1 - (2 / self.robot.num_legs)

    @property
    def stance_start(self) -> Dict[str, float]:
        return self._stance


@dataclass
class WaveGait(Gait):
    """A class to define a wave gait.

    Args:
        robot (Robot): Robot the gait is to be applied to.
        points (int): Total number of joint angles to generate for each leg 
        per gait cycle.
    """

    def __post_init__(self):
        self._stance = {}
        for leg_name in self.robot.leg_names:
            leg_num = int(leg_name[1:])
            if leg_name[0] == 'L':
                # Makes right legs stance first
                leg_num += self.robot.num_legs/2
            self._stance[leg_name] = leg_num * (1 - self.duty_factor)

    @property
    def duty_factor(self) -> float:
        # 1 leg swing at a time
        return 1 - (1 / self.robot.num_legs)

    @property
    def stance_start(self) -> Dict[str, float]:
        return self._stance
