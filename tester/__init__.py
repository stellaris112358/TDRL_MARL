from .tester import Tester, batch
from .cartpole import CartPole_Balance
from .walker import Walker_Stand, Walker_Walk, Walker_Run, Walker_Jump, Walker_Jump_Run
from .cheetah import Cheetah_Run
from .quadruped import Quadruped_Walk, Quadruped_Run
from .spread import Spread_v3


TestDict = {
    "cartpole_balance": CartPole_Balance,
    
    "walker_stand": Walker_Stand,
    "walker_walk": Walker_Walk,
    "walker_run": Walker_Run,
    "walker_jump": Walker_Jump,
    "walker_jump_run": Walker_Jump_Run,
    
    "cheetah_run": Cheetah_Run,
    
    "quadruped_walk": Quadruped_Walk,
    "quadruped_run": Quadruped_Run,
    
    "simple_spread_v3": Spread_v3,
}