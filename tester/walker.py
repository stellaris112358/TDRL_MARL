from tester import Tester, batch
import numpy as np


class Walker_Stand(Tester):
    
    def __init__(self):
        super().__init__()
        self.ds = 24
        self.da = 6
        self._stand_height = 1.2
        self._torsor_angle_limit = [0.9, 1]
        """
        obs:
            0-13: orientations (
                torsor      xz zz
                right_thigh xz zz
                right_leg   xz zz
                right_foot  xz zz
                left_thigh  xz zz
                left_leg    xz zz
                left_foot   xz zz
            )
            14: height
            15-23: velocity (
                rootz
                rootx
                rooty
                right_hip
                right_knee
                right_ankle
                left_hip
                left_knee
                left_ankle
            )
        act:
            0-5: force
        """
        # sort pass-fail tests by the importance
        self._pf_tests = [
            self.pf_height,
            self.pf_upright,
        ]
        
        self._ind_tests = [
            self.ind_height,
            self.ind_upright,
            self.ind_vel,
        ]
    
    @batch
    def pf_upright(self, inputs:np.ndarray, s_nexts:np.ndarray):
        """
        Check if the torsor is always upright
        """
        torsor_angle_cos = s_nexts[:,0]
        if np.sum(torsor_angle_cos > self._torsor_angle_limit[0]) / len(inputs) > 0.8:
            return True
        return False
    
    @batch
    def pf_height(self, inputs:np.ndarray, s_nexts:np.ndarray):
        """
        Check if the walker stand
        """
        height = s_nexts[:,14]
        if np.sum(height > self._stand_height) / len(inputs) > 0.8:
            return True
        return False
    
    @batch
    def ind_upright(self, inputs:np.ndarray, s_nexts:np.ndarray):
        ind = 0
        for i in range(len(s_nexts)):
            torsor_angle_cos = s_nexts[i,0]
            if torsor_angle_cos > self._torsor_angle_limit[0]:
                ind += 1
        return ind
    
    @batch
    def ind_height(self, inputs:np.ndarray, s_nexts:np.ndarray):
        ind = 0
        for i in range(len(s_nexts)):
            height = s_nexts[i,14]
            if height > self._stand_height:
                ind += 1
        return ind
    
    @batch
    def ind_vel(self, inputs:np.ndarray, s_nexts:np.ndarray):
        ind = 0
        x_vel = s_nexts[:,16]
        ind -= np.sum(x_vel ** 2)
        return ind


class Walker_Walk(Walker_Stand):
    def __init__(self):
        super().__init__()
        self._speed = 1.5
        self.epsilon = 0.2
    
        self._pf_tests = [
            self.pf_height,
            self.pf_upright,
            self.pf_speed,
        ]

        self._ind_tests = [
            self.ind_height,
            self.ind_upright,
            self.ind_speed,
        ]
    
    @batch
    def pf_speed(self, inputs:np.ndarray, s_nexts:np.ndarray):
        x_vel = s_nexts[:,16]
        mean_vel = np.mean(x_vel)
        if abs(mean_vel - self._speed) < self.epsilon:
            return True
        return False
    
    @batch
    def ind_speed(self, inputs:np.ndarray, s_nexts:np.ndarray):
        x_vel = s_nexts[:,16]
        mean_vel = np.mean(x_vel)
        return mean_vel


class Walker_Run(Walker_Walk):
    def __init__(self):
        super().__init__()
        self._speed = 8


class Walker_Jump(Walker_Stand):
    def __init__(self):
        super().__init__()
        self._jump_height = 5
        self._pf_tests = [
            self.pf_upright,
            self.pf_jump,
        ]
        
        self._ind_tests = [
            self.ind_upright,
            self.ind_jump,
            self.ind_vel,
        ]
    
    @batch
    def pf_jump(self, inputs:np.ndarray, s_nexts:np.ndarray):
        heights = s_nexts[:,14]
        velocities = s_nexts[:,15]
        if np.max(heights) > self._jump_height:
            return True
        return False
        
    
    @batch
    def ind_jump(self, inputs:np.ndarray, s_nexts:np.ndarray):
        height = s_nexts[:,14]
        return np.max(height)
    
    @batch
    def ind_vel(self, inputs:np.ndarray, s_nexts:np.ndarray):
        ind = 0
        x_vel = s_nexts[:,16]
        ind -= np.sum(x_vel ** 2)
        return ind


class Walker_Jump_Vel(Walker_Jump):
    
    def __init__(self):
        super().__init__()
        self._speed = 1.5
        
        self._pf_tests = [
            self.pf_upright,
            self.pf_jump,
            self.pf_speed,
        ]
        
        self._ind_tests = [
            self.ind_upright,
            self.ind_jump,
            self.ind_speed,
        ]
    
    @batch
    def pf_speed(self, inputs:np.ndarray, s_nexts:np.ndarray):
        x_vel = s_nexts[:,16]
        mean_vel = np.mean(x_vel)
        if mean_vel > self._speed:
            return True
        return False
    
    @batch
    def ind_speed(self, inputs:np.ndarray, s_nexts:np.ndarray):
        x_vel = s_nexts[:,16]
        mean_vel = np.mean(x_vel)
        return mean_vel

class Walker_Jump_Run(Walker_Jump_Vel):
    def __init__(self):
        super().__init__()
        self._speed = 8
