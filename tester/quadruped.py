from tester import Tester, batch
import numpy as np

class Quadruped_Walk(Tester):
    
    def __init__(self):
        super().__init__()
        self.ds = 24
        self.da = 6
        self._stand_height = 1.2
        self._torsor_angle_limit = [0.9, 1]
        self._speed = 0.5
        """
        obs:
            0-43: egocentric_state
            44-46: torso_velocity
            47: torso_upright
            48-53: imu
            54-77: force_torque
        act:
            0-11: force
        """
        # sort pass-fail tests by the importance
        self._pf_tests = [
            self.pf_upright,
            self.pf_speed
        ]
        
        self._ind_tests = [
            self.ind_upright,
            self.ind_speed
        ]
    
    @batch
    def pf_upright(self, inputs:np.ndarray, s_nexts:np.ndarray):
        """
        Check if the torsor is always upright
        """
        torsor_angle_cos = s_nexts[:,47]
        if np.sum(torsor_angle_cos > self._torsor_angle_limit[0]) / len(inputs) > 0.8:
            return True
        return False
    
    @batch
    def pf_speed(self, inputs:np.ndarray, s_nexts:np.ndarray):
        x_vel = s_nexts[:,44]
        speed = np.percentile(x_vel, 15)
        if speed > self._speed:
            return True
        return False
    
    @batch
    def ind_upright(self, inputs:np.ndarray, s_nexts:np.ndarray):
        ind = 0
        for i in range(len(s_nexts)):
            torsor_angle_cos = s_nexts[i,47]
            if torsor_angle_cos > self._torsor_angle_limit[0]:
                ind += 1
        return ind
    
    @batch
    def ind_speed(self, inputs:np.ndarray, s_nexts:np.ndarray):
        x_vel = s_nexts[:,44]
        mean_vel = np.mean(x_vel)
        return mean_vel

class Quadruped_Run(Quadruped_Walk):
    def __init__(self):
        super().__init__()
        self._speed = 5