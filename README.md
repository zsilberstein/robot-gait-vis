# Robot-Gait-Vis

Robot-Gait-Vis is a python package designed to help rapidly visualize the performance of different 
gaits and trajectories for various types of walking robots.

## Features
- Three different types of legs
- Three different types of gaits
- Rapid animation of walking robot
- Tracking and plotting of joint angles over time
- Tracking and plotting of end-effector positions over time

## Installation
To install the package with `pip`:
```python
pip3 install robot-gait-vis
```

## Dependencies
In addition to python version 3.9 or newer, this project depends on the following open source packages:
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

## Basic Usage
The following examples assume that `robot-gaot-vis` has been imported as `rgv`:
```python
import robot_gait_vis as rgv
```
To create a planar leg with segment lengths of 0.5 and 1 meters:
```python
planar_leg = rgv.PlanarLeg((0.5, 1))
```
To create a quadruped robot with body dimensions of 0.5 by 0.25 by 0.5 meters:
```python
quadruped = rgv.Robot((0.5, 0.25, 0.5), planar_leg, 4)
```

## Detailed Usage
The example.py file illustrates detailed example usage of the package. The example file generates walking motion for a simple biped robot as well as a hexapod robot and outputs the GIFs shown below.
<p float="left">
    <img src="https://github.com/zsilberstein/robot-gait-vis/blob/master/biped_ex.gif?raw=true" alt="Gif of a simple biped robot walking" width="365" height="274" />
    <img src="https://github.com/zsilberstein/robot-gait-vis/blob/master/hexapod_ex.gif?raw=true" alt="Gif of a hexapod robot walking" width="365" height="274" />
</p>

## Testing
To run tests for the project:
```python
python3 -m unittest
```

## Future Improvements
- Add new leg types
- Add new gaits
- Add new end effector trajectories
- Add more animation options  
### Have your own improvement in mind? 
Feel free to fork the project and then put in a pull request with the desired improvement. Please be sure to include relevant tests in any pull request.

## License
This project is licensed under the [MIT License](https://github.com/zsilberstein/robot-gait-vis/blob/master/LICENSE). See the LICENSE page for more information.

## Author
**Zach Silberstein**  
Email: **zach.silberstein@gmail.com**   
Github: [@zsilberstein](https://github.com/zsilberstein)