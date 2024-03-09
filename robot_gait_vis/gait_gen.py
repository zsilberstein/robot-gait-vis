from typing import Tuple, List, Dict
import numpy as np

from robot_gait_vis.leg import Leg
from robot_gait_vis.robot import Robot


def create_line(
        leg: Leg,
        init_thetas: List[float],
        pos_change: Tuple[float, float, float],
        num_points: int
) -> List[List[float]]:
    """Generates a sequence of joint angles to move the end effector of the 
    leg along the specified straight line.

    Args:
        leg (Leg): Leg that is to follow the path.
        init_thetas (List[float]): Starting joint angles of the leg.
        pos_change (Tuple[float, float, float]): Specifies how far to move 
        the leg in (X, Y, Z).
        num_points (int): Number of points on the path the leg is to pass 
        through not including the initial point defined by 
            init_thetas. Must be positive.

    Returns:
        List[List[float]]: List of length num_points + 1. Each index stores a 
        leg configuration along the calculated path starting with 
            the init_thetas.
    """
    # Determine how far to move in each direction at each point
    deltas = [x / num_points for x in pos_change]

    init_pos = leg.forward_kinematics(init_thetas)[-1]

    # Find (X,Y,Z) of each point on the line
    points = [np.add(init_pos, np.multiply(deltas, i))
              for i in range(num_points+1)]

    thetas = [leg.inverse_kinematics(point)[0] for point in points]

    return thetas


def create_semi_ellipse(
        leg: Leg,
        init_thetas: List[float],
        base: Tuple[float, float, float],
        height: Tuple[float, float, float],
        num_points: int
) -> List[List[float]]:
    """Generates a sequence of joint angles to move the end effector of the 
    leg along the specified semi-ellipse.

    Args:
        leg (Leg): Leg that is to follow the path.
        init_thetas (List[float]): Starting joint angles of leg.
        base (Tuple[float, float, float]): Vector from the end-effector 
        position with init_thetas to the desired final position of the 
            end-effector.
        height (Tuple[float, float, float]): Vector from the end-effector 
        position with init_thetas to point of ellipse with max 
            curvature.
        num_points (int): Number of points on the path the leg is to pass 
        through not including the initial point defined by 
            init_thetas. Must be positive.
    Returns:
        List[List[float]]: List of length num_points + 1. Each index stores a 
        leg configuration along the calculated path starting with 
            the init_thetas.
    """
    init_pos = leg.forward_kinematics(init_thetas)[-1]
    center = np.add(init_pos, np.divide(base, 2))

    # From center to init_pos
    u = -1 * np.divide(base, 2)

    # From center to raise
    v = np.array(height) - center + init_pos

    ts = np.linspace(0, np.pi, num_points+1)
    points = [center + u*np.cos(t) + v*np.sin(t) for t in ts]
    thetas = [leg.inverse_kinematics(point)[0] for point in points]

    return thetas


def create_ellipse_trajectory(
        robot: Robot,
        init_thetas: Dict[str, List[float]],
        right_stance_vec: Tuple[float, float, float],
        right_raise_height: Tuple[float, float, float],
        left_stance_vec: Tuple[float, float, float],
        left_raise_height: Tuple[float, float, float],
        num_points: int,
        duty_factor: float
) -> Dict[str, Dict[str, List[float]]]:
    """Generates an elliptical swing trajectory and a linear 
    stance trajectory for each leg on the robot.

    Args:
        robot (Robot): Robot to generate leg trajectories for.
        init_thetas (Dict[str, List[float]]): Dictionary specifying the joint 
        angles at the start of stance for each leg.
        right_stance_vec (Tuple[float, float, float]): Specifies how far to 
        move each right leg in (X,Y,Z) during stance in the world 
            frame.
        right_raise_height (Tuple[float, float, float]): Vector in world frame 
        from the end of stance position to point of swing ellipse 
            with max curvature for each right leg.
        left_stance_vec (Tuple[float, float, float]): Specifies how far to 
        move each left leg in (X,Y,Z) during stance in the world frame.
        left_raise_height (Tuple[float, float, float]): Vector in world frame 
        from the end of stance position to point of swing ellipse 
            with max curvature for each left leg.
        num_points (int): Total number of points to generate for each leg. 
        Must be greater than 1.
        duty_factor (float): Fraction of total cycle time stance occurs for 
        each leg.

    Returns:
        Dict[str, Dict[str, List[float]]]: Returns a dict for each leg 
        specifying the joint angles along the calculated path 
            for 'stance' and 'swing'.
    """

    # Determine how many points should be generated during each phase
    points_stance = int(num_points * duty_factor)
    points_swing = num_points - points_stance

    trajectories = {}
    for leg_name in robot.leg_names:
        thetas = {}

        # If leg is on right side
        if leg_name[0] == 'R':
            # Legs swing in direction of travel, opposite direction of stance
            right_swing_base = np.multiply(-1, right_stance_vec)
            thetas['stance'] = create_line(robot.leg_type,
                                           init_thetas[leg_name],
                                           right_stance_vec,
                                           points_stance)
            thetas['swing'] = create_semi_ellipse(robot.leg_type,
                                                  thetas['stance'][-1],
                                                  right_swing_base,
                                                  right_raise_height,
                                                  points_swing)

        else:
            left_swing_base = np.multiply(-1, left_stance_vec)
            thetas['stance'] = create_line(robot.leg_type,
                                           init_thetas[leg_name],
                                           left_stance_vec,
                                           points_stance)
            thetas['swing'] = create_semi_ellipse(robot.leg_type,
                                                  thetas['stance'][-1],
                                                  left_swing_base,
                                                  left_raise_height,
                                                  points_swing)

        trajectories[leg_name] = thetas

    return trajectories
