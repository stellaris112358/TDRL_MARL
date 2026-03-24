from typing import List, Callable
from functools import wraps
import numpy as np
from collections import OrderedDict

def hashable_obj(obj):
    """convert object to hashable object (bytes)"""
    if isinstance(obj, np.ndarray):
        return obj.tobytes()
    else:
        return obj

def cached_single_call(func, max_size=2048):
    cache = OrderedDict()
    @wraps(func)
    def wrapper(self, input, s_next):
        key = (hashable_obj(input), hashable_obj(s_next))  # s_next should be hashable
        if key in cache:
            cache.move_to_end(key)  # move to end to mark as recently used
            return cache[key]
        result = func(self, input, s_next)
        cache[key] = result
        if len(cache) > max_size:
            cache.popitem(last=False)  # remove the least recently used item
        return result
    return wrapper

def batch(func):
    cached_func = cached_single_call(func)
    @wraps(func)
    def wrapper(self, inputs, s_nexts):
        if len(inputs[0].shape) > 1:
            results = []
            for input, s_next in zip(inputs, s_nexts):
                results.append(cached_func(self, input, s_next))
            return np.array(results)
        else:
            return np.array(cached_func(self, inputs, s_nexts))
    return wrapper

class Tester(object):
    def __init__(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        self.da = 0 # dimension of state-action space
        self.ds = 0 # dimension of state space
        self._pf_tests = [] # list of pass-fail tests
        self._ind_tests = []    # list of indicative trajectory tests

    def batch_test(self, inputs:List, s_nexts:List, test_set:List[Callable]):
        results = []
        for test in test_set:
            results.append(test(inputs, s_nexts))
        results = np.moveaxis(np.array(results), 0, -1)
        return results
    
    @batch
    def pf_example(self, inputs:np.ndarray, s_nexts:np.ndarray):
        """
        Example pass-fail test that always returns False.
        This is just a placeholder and should be replaced with actual tests.
        A tester could contains multiple pass-fail tests.
        Each test should return a single boolean value that indicates whether the agent passes the test.
        inputs: np.ndarray of shape (trajectory length, dsa)
        s_nexts: np.ndarray of shape (trajectory length, ds)
        """
        return False  # Replace with actual logic
    
    @batch
    def ind_example(self, inputs:np.ndarray, s_nexts:np.ndarray):
        """
        Example indicative trajectory test that always returns 0.
        This is just a placeholder and should be replaced with actual tests.
        A tester could contains multiple indicative trajectory tests.
        Each test should return a single value that indicates the agent's performance in a specific aspect .
        inputs: np.ndarray of shape (trajectory length, dsa)
        s_nexts: np.ndarray of shape (trajectory length, ds)
        """
        return 0  # Replace with actual logic
    
    def pf_test(self, inputs, s_nexts):
        """Perform pass-fail test"""
        return self.batch_test(inputs, s_nexts, self._pf_tests)
    
    def ind_test(self, inputs, s_nexts):
        """Perform indicative trajectory test"""
        return self.batch_test(inputs, s_nexts, self._ind_tests)
    
    @property
    def dsa(self):
        return self.ds + self.da
    

