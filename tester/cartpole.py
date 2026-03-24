from tester import Tester, batch
import numpy as np

class CartPole_Balance(Tester):
    
    def __init__(self):
        super().__init__()
        self.ds = 5
        self.da = 1
        self.cart_pos_limit = [-0.25, 0.25]
        self.pole_angle_cos_limit = [0.995, 1]
        """
        obs:
            0: cart position
            1: pole angle cos
            2: pole angle sin
            3: cart velocity
            4: pole angular velocity
        act:
            0: force
        """
        # sort pass-fail tests by the importance
        self._pf_tests = [
            self.pf_upright,
            self.pf_pos
        ]
        
        self._ind_tests = [
            self.ind_upright,
            self.ind_pos
        ]
    
    @batch
    def pf_upright(self, inputs:np.ndarray, s_nexts:np.ndarray):
        """
        Check if the cart is always upright
        """
        pole_angle_cos = s_nexts[:,1]
        return np.all(pole_angle_cos > self.pole_angle_cos_limit[0])
    
    @batch
    def pf_pos(self, inputs:np.ndarray, s_nexts:np.ndarray):
        """
        Check if the cart is always in the middle
        """
        cart_pos = s_nexts[:,0]
        return np.all(np.logical_and(cart_pos > self.cart_pos_limit[0], cart_pos < self.cart_pos_limit[1]))
    
    @batch
    def ind_upright(self, inputs:np.ndarray, s_nexts:np.ndarray):
        ind = 0
        for i in range(len(s_nexts)):
            pole_angle_cos = s_nexts[i,1]
            if pole_angle_cos > self.pole_angle_cos_limit[0]:
                ind += 1
        return ind
    
    @batch
    def ind_pos(self, inputs:np.ndarray, s_nexts:np.ndarray):
        ind = 0
        for i in range(len(s_nexts)):
            cart_pos = s_nexts[i,0]
            if np.logical_and(cart_pos > self.cart_pos_limit[0], cart_pos < self.cart_pos_limit[1]):
                ind += 1
        return ind


if __name__ == "__main__":
    tester = CartPole_Balance()