from tester import Tester, batch
import numpy as np

class Cheetah_Run(Tester):
    
    def __init__(self):
        super().__init__()
        self.ds = 17
        self.da = 6
        
        self._run_speed = 10
        
        """
        obs:
            0-7: qpos(
                rootz
                rooty
                bthigh
                bshin
                bfoot
                fthigh
                fshin
                ffoot
            )
            8-16: qvel(
                rootx
                rootz
                rooty
                bthigh
                bshin
                bfoot
                fthigh
                fshin
                ffoot
            )
            
        """
        
        self._pf_tests = [
            self.pf_speed,
        ]
        
        self._ind_tests = [
            self.ind_speed,
        ]
    
    @batch
    def pf_speed(self, inputs, s_nexts):
        """test mean speed"""
        speeds = s_nexts[:, 8]
        return np.mean(speeds) >= self._run_speed
    
    @batch
    def ind_speed(self, inputs, s_nexts):
        speeds = s_nexts[:, 8]
        return np.mean(speeds)
        