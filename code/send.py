import numpy as np
from scipy.stats import norm
from scipy import integrate
from scipy import signal
import matplotlib.pyplot as plt

class CPMGenerator:
    def __init__(self, a, Es=1, h=0.5, T=1, BT=0.5, Lg=2, M=2, K=1, fs=10):
        self.h = h
        self.Es = Es
        self.T = T        
        self.BT = BT
        self.B = BT / T
        self.Lg = Lg
        self.LT = Lg * T
        self.Q = 2 ** (Lg-1)
        self.a = a
        self.M = M
        self.K = K
        self.fs = fs
        self.P = int(np.log2(M))

    def DFreqResponse(self):
        raise NotImplementedError
    
    def DPhaseResponse(self):
        g = self.DFreqResponse()
        q = np.zeros_like(g)
        # trapz method, maybe Simpon can be better
        q[1:] = (g[:-1] + g[1:]) / (self.fs * 2)
        q = np.cumsum(q)
        q = q / q[-1] * 0.5
        return q
    
    def Duration(self):
        D = []
        for k in range(self.Q):
            if k == 0:
                D.append(int(self.Lg + 1))
            else:
                D.append(int(self.Lg - 1 - np.floor(np.log2(k))))
        return D
    
    def Du(self):
        q = self.DPhaseResponse()
        u = np.zeros([self.P, 2*len(q)-1])
        for l in range(self.P):
            h = 2**l * self.h
            u[l][0:len(q)] = np.sin(2*h*np.pi*q)/np.sin(h*np.pi)
            u[l][len(q)-1::1] = u[l][len(q)-1::-1]
        return u

    def beta(self, k, i):
        if i == 0:
            return 0
        elif i > 0:
            return (k >> (i-1)) & 1 # return ith bit in the radix-2 representation of k
        else:
            raise ValueError('i can not be negative')

    def Dpulse(self):
        D = self.Duration()
        u = self.Du()
        pulse = [np.zeros([self.P, d*self.T*self.fs]) for d in D]
        for k in range(self.Q):
            for l in range(self.P):
                for i in range(D[k]*self.fs):
                    tmp = 1
                    for j in range(self.Lg):
                        idx = i + j*self.fs + self.beta(k, j)*self.Lg*self.fs
                        if idx >= len(u[l]):
                            tmp = 0
                            break
                        tmp *= u[l][idx]
                    pulse[k][l][i] = tmp
        return pulse

    def DpulseChart(self):
        pulse = self.Dpulse()
        if self.M == 2:
            g = []
            for item in pulse:
                g.append(item[0])
        elif self.M == 4 and self.Lg == 2:
            g = np.zeros([5, (self.Lg+1)*self.fs])
            g[0, :] = pulse[0][0, :] * pulse[0][1, :]
            g[1, 0:self.Lg*self.fs] = pulse[0][0, self.fs:] * pulse[0][1, 0:self.Lg*self.fs]
            g[2, 0:self.Lg*self.fs] = pulse[0][0, 0:self.Lg*self.fs] * pulse[0][1, self.fs:]
            g[3, 0:self.fs] = pulse[0][0, self.Lg*self.fs:] * pulse[0][1, 0:self.fs]
            g[4, 0:self.fs] = pulse[0][0, 0:self.fs] * pulse[0][1, self.Lg*self.fs:]
            """ 
            g1 and g2 are quite similar. They can be replaced by an average one.
            for more details, see Noncoherent Sequence Detection of Continuous Phase Modulations
            """
            g[1, :] = (g[1, :]+g[2, :]) / 2
            g[2, :] = (g[3, :]+g[4, :]) / 2
        else:
            raise NotImplementedError
        return g

    def pseudo(self, k, n):
        raise NotImplementedError
    
    def Dsymbols(self):
        q = self.DPhaseResponse()
        a_len = len(self.a)
        q_len = len(q)
        s_len = (a_len+self.Lg)*self.fs
        s = np.zeros(s_len, dtype=complex)
        for i in range(a_len):
            l = i * self.fs
            r = l + q_len
            s[l:r] += self.a[i]*q
            s[r:] += self.a[i] / 2
        s = np.sqrt(2*self.Es/self.T) * np.exp(1j*2*np.pi*self.h*s)
        return s


class GMSKGenerator(CPMGenerator):
    def __init__(self, *args):
        super(GMSKGenerator, self).__init__(*args)

    def DFreqResponse(self):
        t = np.linspace(-self.Lg*self.T/2, self.Lg*self.T/2, self.Lg*self.T*self.fs+1)
        g = 1/2/self.T*(norm.cdf(2*np.pi*self.B*(t+self.T/2)/np.log(2)**0.5)-norm.cdf(2*np.pi*self.B*(t-self.T/2)/np.log(2)**0.5))
        return g


class RCGenerator(CPMGenerator):
    def __init__(self, *args):
        super(RCGenerator, self).__init__(*args)

    def DFreqResponse(self):
        t = np.linspace(0, self.Lg*self.T, self.Lg*self.T*self.fs+1)
        g = 1/(2*self.Lg*self.T)*(t-np.cos(2*np.pi*t/(self.Lg*self.T)))
        return g
    
    # this overwrite is not necessary, but for 2RC signal, phase response can be directly written
    def DPhaseResponse(self):
        t = np.linspace(0, self.Lg*self.T, self.Lg*self.T*self.fs+1)
        q = 1/(2*self.Lg*self.T)*(t-self.Lg*self.T/(2*np.pi)*np.sin(2*np.pi*t/self.Lg))
        return q


    # these are continuous versions, which are too slow.

    # def gamma(self):
    #     a_len = len(self.a)
    #     a_hat = (self.a + 2**self.P - 1) / 2
    #     gamma_hat = np.zeros([a_len, self.P])
    #     for l in range(self.P):
    #         gamma_hat[:, l] = self.beta(self.a, l+1)
    #     gamma = 2*gamma_hat - 1
    #     return gamma


    # def FreqResponse(self, t):
    #     g = 1/2/self.T*(norm.cdf(2*np.pi*self.B*(t-self.LT/2+self.T/2)/np.log(2)**0.5)-norm.cdf(2*np.pi*self.B*(t-self.LT/2-self.T/2)/np.log(2)**0.5))

    #     return g


    # def pulse(self, k, t):
    #     u_list = [self.u(t+i*self.T+self.beta(k, i)*self.LT) for i in range(self.Lg)]

    #     return np.prod(u_list)


    # def u(self, t):
    #     if t >=0 and t <= self.LT:
    #         return np.sin(2*self.h*np.pi*self.PhaseResponse(t)) / np.sin(self.h*np.pi)
    #     elif t > self.LT and t <= self.LT*2:
    #         return self.u(2*self.LT-t)
    #     else:
    #         return 0


    # def symbols(self, t):
    #     s = 0
    #     t_floor = int(t)
    #     for i in range(self.Lg):
    #         if t_floor - i < 0 or t_floor - i >= len(self.a):
    #             continue
    #         else:
    #             s += self.a[t_floor-i] * self.PhaseResponse(t-t_floor+i*self.T)
    #     return np.exp(1j*2*np.pi*self.h*s)


    # def PhaseResponse(self, t):
    #     if t <= 0:
    #         return 0
    #     elif t >= self.LT:
    #         return 0.5
    #     else:
    #         C, _ = integrate.quad(self.FreqResponse, 0, self.LT)   
    #         q, _ = integrate.quad(self.FreqResponse, 0, t) 
    #         q = q / C * 0.5 
    #         return q