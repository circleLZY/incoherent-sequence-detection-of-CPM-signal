import numpy as np
import matplotlib.pyplot as plt
from send import *
from transmit import *
from receive import *

class Configs(object):
    s_len = 100000
    BT = 0.25
    h = 0.5
    K = 1    
    T = 1
    Es = 1
    Lg = 2
    fs = 10
    BT = 0.25
    M = 2
    L = 2

def getBER(send, receive):
    return np.sum(send != receive) / len(send)

def MaryGenerator(M, s_len):
    rand = np.random.rand(s_len)
    return np.round(rand*(M-1))*2-M+1

def GMSKSimulation():
    N = [2,3,4,5,5,5]
    S = [2,4,4,4,8,32]
    N0 = [10**(-n/20) for n in range(20)]
    BER = np.zeros([6, 20])
    SNR = np.arange(20)

    for i in range(6):
        for j in range(20):
            alpha = MaryGenerator(Configs.M, Configs.s_len)
            GMSK = GMSKGenerator(alpha, Configs.Es, Configs.h, Configs.T, Configs.BT, Configs.Lg, Configs.M, Configs.K, Configs.fs)
            s = GMSK.Dsymbols()
            h = GMSK.DpulseChart()
            duration = GMSK.Duration()

            channel = Transmitter(s, N0[j], Configs.fs)
            r = channel.AWGN()

            reciver = GMSKReciver(r, h, Configs.T, Configs.L, N[i], S[i], Configs.fs, Configs.s_len, duration, Configs.K, Configs.h, Configs.M)
            alpha_hat = reciver.ViterbiDemodulation()
            BER[i][j] = getBER(alpha, alpha_hat)

    GMSKdraw(SNR, BER)

def GMSKdraw(SNR, BER):
    plt.figure()
    plt.plot(SNR, BER[0], color='black', linestyle='-', marker='>', linewidth=0.5)
    plt.plot(SNR, BER[1], color='black', linestyle='-', marker='^', linewidth=0.5)
    plt.plot(SNR, BER[2], color='black', linestyle='-', marker='<', linewidth=0.5)
    plt.plot(SNR, BER[3], color='black', linestyle='-', marker='d', linewidth=0.5)
    plt.plot(SNR, BER[4], color='black', linestyle='-', marker='s', linewidth=0.5)
    plt.plot(SNR, BER[5], color='black', linestyle='-', marker='o', linewidth=0.5)

    plt.xlabel('Eb/n0   [dB]')
    plt.ylabel('BER')
    plt.yscale("log")
    plt.legend(["N=2;S=2","N=3;S=4","N=4;S=4","N=5;S=4","N=5;S=8","N=5;S=32"])
    plt.grid(which="both",linestyle = '--',linewidth = 0.2)
    plt.show()

def RCSimulation():
    N = [3,3,4,4]
    S = [4,16,16,64]
    N0 = [10**(-n/20) for n in range(20)]
    BER = np.zeros([4, 20])
    SNR = np.arange(20)

    for i in range(4):
        for j in range(20):
            alpha = MaryGenerator(Configs.M, Configs.s_len)
            RC = RCGenerator(alpha, Configs.Es, Configs.h, Configs.T, Configs.BT, Configs.Lg, Configs.M, Configs.K, Configs.fs)
            s = RC.Dsymbols()
            h = RC.DpulseChart()
            duration = RC.Duration()

            channel = Transmitter(s, N0[j], Configs.fs)
            r = channel.AWGN()

            reciver = RCReciver(r, h, Configs.T, Configs.L, N[i], S[i], Configs.fs, Configs.s_len, duration, Configs.K, Configs.h, Configs.M)
            alpha_hat = reciver.ViterbiDemodulation()
            BER[i][j] = getBER(alpha, alpha_hat)

    RCdraw(SNR, BER)

def RCdraw(SNR, BER):
    plt.figure()
    plt.plot(SNR, BER[0], color='black', linestyle='-', marker='>', linewidth=0.5)
    plt.plot(SNR, BER[1], color='black', linestyle='-', marker='^', linewidth=0.5)
    plt.plot(SNR, BER[2], color='black', linestyle='-', marker='<', linewidth=0.5)
    plt.plot(SNR, BER[3], color='black', linestyle='-', marker='d', linewidth=0.5)

    plt.xlabel('Eb/n0   [dB]')
    plt.ylabel('BER')
    plt.yscale("log")
    plt.legend(["N=3;S=4","N=3;S=16","N=4;S=16","N=4;S=64"])
    plt.grid(which="both",linestyle = '--',linewidth = 0.2)
    plt.show()

if __name__ == '__main__':
    GMSKSimulation()

    Configs.K = 2
    GMSKSimulation()
    
    Configs.M = 4
    Configs.h = 0.25
    Configs.L = 1
    Configs.Es = 2
    RCSimulation()

    Configs.K = 3
    RCSimulation()
    
