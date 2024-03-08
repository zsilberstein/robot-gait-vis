from typing import Tuple, List
import numpy as np

from robot_gait_vis.leg import Leg


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
