import numpy as np

class Transmitter:
    def __init__(self, s, N0, fs):
        self.s = s
        self.N0 = N0
        self.fs = fs
    
    def AWGN(self):
        sigma2 = self.N0 * self.fs
        w = [complex(np.random.normal(0, sigma2), np.random.normal(0, sigma2)) for _ in range(len(self.s))]
        theta = np.random.uniform(0, 2*np.pi)
        r = self.s * np.exp(1j*theta) + w
        return r