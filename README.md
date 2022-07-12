# Robot Gym

Provides a modular way of defining learning tasks on the UR10 as gym environments and an abstraction of simulation and reality to enable seamless switching.

## Setup

Change into the root directory of this project and run:
```bash
pip install .
```

## Usage

To define a new task, implement a class that inherits from `BaseTask`.
`BaseTask` expects three mandatory arguments: a sequence of controllers, a sequence of sensors and a reward function.


### Controllers

Controllers are called in everz step (by `BaseTask`) with an arbitrary number of (arbitrarily-dimensional) action vectors as input and execute some action on the environment based on these vectors.
Upon initialization, each controller specifies the names, dimensions and limits of its action vectors.
This information will then be used by the `BaseTask` to define the action space of the environment.
All controllers must inherit from the `robot_gym.core.controllers.Controller` class, which takes care of the normalization of the action vectors.
Examples of controllers can be found in `robot_gym/core/controllers`.

### Sensors

Sensors are called in every step (by `BaseTask`) and return an arbitrary number of (arbitrarily-dimensional) observation vectors.
Similar to the controllers, sensors specify their names, dimensions and limits upon initialization, and `BaseTask` uses this information to define the observation space of the environment.
All sensors must inherit from the `robot_gym.core.sensors.Sensor` class.
Continuous sensors can inherit from `robot_gym.core.sensors.ContinuousSensor`, which will take care of observation vector normalization.
Find examples of sensors in `robot_gym/core/sensors`.


### Reward

The reward function is called in every step (by `BaseTask`) and returns a scalar value indicating the reward of this step.
Each reward function must inherit from `robot_gym.core.rewards.Reward`, which provides reward functions with basic arithmetic functions, making combining them easier.
Find examples of rewards in `robot_gym/core/rewards`.


### Final notes

For an example task, see `robot_gym/core/reacher_task.py`, which implements a simple reaching task on the UR10.
Note that `BaseTask` will structure both the action space and the observation space as a dictionary.
To obtain an environment that uses boxes for both, wrap the `BaseTask` instance in a `robot_gym.core.wrappers.FlattenWrapper`.

Jan Schneider (Jan.Schneider@tuebingen.mpg.de) & Tim Schneider (schneider@ias.tu-darmstadt.de)
