# Box-World
A simple implementation of the environment in Google Deepmind's paper "Relational Deep Reinforcement Learning".
No rendering, only for computer training and testing.

## Metadata
kwargs of `gym.make`

|metadata|description|default|
|-|-|-|
|max_length|the number of boxes in the path to the goal|4|
|max_branch_num|the number of distractor branches|4|
|branch_length| the length of the distractor branches|1|

## Observation
Type: ndarray(14, 14, 3)
|Max|Min|
|-|-|
|0.0|255.0|

|Type|Color|
|-|-|
|Wall|RGB(0.0, 0.0, 0.0)|
|Space|RGB(192.25, 191.25, 191.25)|
|Agent|RGB(127.5, 127.5, 127.5)|
|Gem |RGB(255.0, 255.0, 255.0)|

## Action
Type: Discrete(4)
|Num|Action|
|-|-|
|0|Agent move up|
|1|Agent move left|
|2|Agent move down|
|3|Agent move right|

## Reward
|action|reward|
|-|-|
|collecting the gem|10|
|opening a box in the solution path|1|
|opening a distractor box|-1|
|otherwise|0|

## Termination
Terminated when:
- the gem is collected
- a distractor box is opened
