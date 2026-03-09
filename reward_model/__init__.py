try:
    from .reward_model import RewardModel
    from .reward_model_return import ReturnRewardModel
    from .reward_model_tdrl import TdRLRewardModel
except ImportError:
    pass

from .reward_model_ma_tdrl import MATdRLRewardModel