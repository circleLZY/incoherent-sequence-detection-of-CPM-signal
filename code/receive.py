import numpy as np
from scipy import signal

class NonCoherentReceiver:
    def __init__(self, r, pulse, T, L, N, S, fs, s_len, duration, K, h, M):
        self.L = L
        self.N = N
        self.S = S
        self.pulse = pulse
        self.r = r
        self.fs = fs
        self.s_len = s_len
        self.duration = duration
        self.T = T
        self.K = K
        self.h = h
        self.M = M

    def PulseMatchedFilter(self):
        x = np.zeros([self.K, self.s_len], dtype=complex)
        for k in range(self.K):
            xs = 1 / self.fs * np.convolve(self.pulse[k][::-1], self.r)
            for i in range(self.s_len):
                x[k, i] = xs[i*self.fs+self.fs-1]
        return x

    def WhitenedMatchedFilter(self, x):
        raise NotImplementedError
    
    def ViterbiDemodulation(self):
        raise NotImplementedError

    def Decode(self, idx, path):
        if self.M == 2:
            decode_dict = {0:-1, 1:1}
        elif self.M == 4:
            decode_dict = {0:-3, 1:-1, 2:1, 3:3}
        else:
            raise NotImplementedError
        decode_seq = np.zeros(self.s_len)
        for n in range(self.s_len):
            decode_seq[self.s_len-1-n] = decode_dict[idx%self.M]
            idx = path[int(idx),self.s_len-1-n]
        return decode_seq

class GMSKReciver(NonCoherentReceiver):
    def __init__(self, *args):
        super(GMSKReciver, self).__init__(*args)

    def WhitenedMatchedFilter(self, x):
        if self.K == 1:
            g = np.zeros(5)
            g[2] = 1/self.fs*np.dot(self.pulse[0], self.pulse[0])
            g[1] = 1/self.fs*np.dot(self.pulse[0][0:2*self.fs], self.pulse[0][self.fs:3*self.fs])
            g[0] = 1/self.fs*np.dot(self.pulse[0][0:self.fs], self.pulse[0][2*self.fs:3*self.fs])
            g[3] = g[1]
            g[4] = g[0]
            poles = np.roots(g)
            amp = np.sum(g)/((1-poles[2])*(1-poles[3])*(1-poles[2])*(1-poles[3]))
            a = np.sqrt(amp)*np.convolve([1,-poles[2]],[1,-poles[3]])
            F = a[::-1]
            b = np.array([1])
            z = signal.lfilter(b,a,x[0,:])
        elif self.K == 2:
            F = np.zeros([2,2,3])
            F[:,:,0] = np.array([0.08024,0.2305,0.0263,0.0421]).reshape(2,2)
            F[:,:,1] = np.array([0.6558,0.45977,0,0]).reshape(2,2)
            F[:,:,2] = np.array([0.4442,0,0,0]).reshape(2,2)
            WF00 = np.array([0.0421])
            WF01 = np.array([0,-0.45977,-0.2305])
            WF10 = np.array([-0.0263])
            WF11 = np.array([0.4442,0.6558,0.0824])
            WF0 = np.array([0.0187,0.0155,-0.0027])
            z0 = signal.lfilter(WF00,WF0,x[0,:])+signal.lfilter(WF01,WF0,x[1,:])
            z1 = signal.lfilter(WF10,WF0,x[0,:])+signal.lfilter(WF11,WF0,x[1,:])
            z = np.zeros([2,self.s_len],dtype = complex)
            z = np.array([z0,z1])
        else:
            raise NotImplementedError
        
        return z, F
    
    def ViterbiDemodulation(self):
        x = self.PulseMatchedFilter()
        z, F = self.WhitenedMatchedFilter(x)
        path = np.zeros([self.S, self.s_len])
        depth = np.log2(self.S)
        cost = np.zeros(self.S)
        Lambda = np.zeros([self.S,2])
        operator = np.array([-1j, 1j])
        state = np.arange(self.S)
        if self.K == 1:
            Y = np.zeros([self.N, self.S, 2], dtype=complex)
            alpha = np.zeros([self.S, self.N+self.L], dtype = complex)
            alpha_temp = np.zeros([self.S, 2, self.N+self.L], dtype = complex)   
            for n in range(self.s_len):
                if n < depth:    
                    if n == 0:
                        alpha_temp[0, :, 0] = operator
                        Y[0, 0, :] = F[0] * alpha_temp[0, :, 0]
                        Lambda[0, :] = abs(z[0]*Y[0, 0, :].conjugate()) - 0.5*(abs(Y[0, 0, :])**2)

                        cost[0:2] = Lambda[0, :]
                        path[0:2, 0] = 0
                        alpha[0:2, :] = alpha_temp[0, :, :]
                    else: 
                        index = state[0:2**n]
                        alpha_temp[index, :, 0] = operator * alpha[index, 0:1]
                        alpha_temp[index, :, 1:] = np.stack([alpha[index, :-1]]*2, axis=1)
                        for i in range(self.N):
                            tmp = 0
                            for l in range(self.L+1):
                                tmp += F[l] * alpha_temp[index, :, i+l]
                            Y[i, index, :] = tmp
                        tmp = 0
                        for i in range(min(self.N-1,n-1)):
                            tmp += z[n-i-1] * Y[i+1, index, :].conjugate()
                        Lambda[index, :] = abs(tmp + z[n]*Y[0, index, :].conjugate()) - abs(tmp) - 0.5*(abs(Y[0, index, :])**2)
                        
                        state1 = state[0:2**(n+1)] // 2
                        index = state[0:2**(n+1)]
                        cost[index] = cost[state1] + Lambda[state1, state[index%2]]
                        path[index, n] = state1
                        alpha[index, :] = alpha_temp[state1, state[index%2], :]

                else:          
                    alpha_temp[:, :, 0] = operator * alpha[:, 0:1]
                    alpha_temp[:, :, 1:] = np.stack([alpha[:, :-1]]*2, axis=1)
                    for i in range(self.N):
                        tmp = 0
                        for l in range(self.L+1):
                            tmp += F[l] * alpha_temp[:, :, i+l]
                        Y[i] = tmp
                    tmp = 0
                    for i in range(min(self.N-1,n-1)):
                        tmp += z[n-i-1] * Y[i+1].conjugate()
                    Lambda = abs(tmp + z[n]*Y[0].conjugate()) - abs(tmp) - 0.5*(abs(Y[0])**2)

                    state1 = state // 2
                    state2 = state1 + self.S//2
                    cost1 = cost[state1] + Lambda[state1, state%2]
                    cost2 = cost[state2] + Lambda[state2, state%2]
                    index = cost1 < cost2
                    cost = cost1
                    cost[np.where(index)] = cost2[index]
                    path[:, n] = state1
                    path[:, n][np.where(index)] = state2[index]
                    alpha[:, :] = alpha_temp[state1, state%2, :]
                    alpha[:, :][np.where(index)] = alpha_temp[state2, state%2, :][index]
        elif self.K == 2:
            Y = np.zeros([self.N, self.S, 2, 2], dtype=complex)
            alpha = np.zeros([self.S, 2, self.N+self.L], dtype = complex)
            alpha_temp = np.zeros([self.S,2,2,self.N+self.L],dtype = complex)
            for n in range(self.s_len):
                if n < depth:
                    if n == 0:
                        alpha_temp[0, :, 0, 0] = operator
                        alpha_temp[0, :, 1, 0] = operator
                        Y[0, 0, :, 0] = np.matmul(F[:,:,0].T,alpha_temp[0,0,:,0])
                        Y[1, 0, :, 1] = np.matmul(F[:,:,1].T,alpha_temp[0,1,:,0])
                        Lambda[0, :] = abs(z[0][0]*Y[0, 0, 0, :].conjugate() + z[1][0]*Y[0, 0, 1, :].conjugate())
                        
                        cost[0:2] = Lambda[0, :]
                        path[0:2, 0] = 0
                        alpha[0:2, :, :] = alpha_temp[0, :, :, :]
                    else:
                        index = state[0:2**n]
                        alpha_temp[index, 0, 0, 0] = operator[0] * alpha[index, 0, 0]
                        alpha_temp[index, 1, 0, 0] = operator[1] * alpha[index, 0, 0]
                        if n == 1:
                            alpha_temp[index, 0, 1, 0] = operator[0] * complex(0, 1)
                            alpha_temp[index, 1, 1, 0] = operator[1] * complex(0, 1)
                        else:
                            alpha_temp[index, 0, 1, 0] = operator[0] * alpha[index, 0, 1]
                            alpha_temp[index, 1, 1, 0] = operator[1] * alpha[index, 0, 1]
                        alpha_temp[index, :, :, 1:] = np.stack([alpha[index, :, :-1]]*2, axis=1)
                        for i in range(self.N):
                            for s in range(2**n):
                                tmp1 = 0
                                tmp2 = 0
                                for l in range(self.L+1):
                                    tmp1 += np.matmul(F[:,:,l].T,alpha_temp[s,0,:,i+l])
                                    tmp2 += np.matmul(F[:,:,l].T,alpha_temp[s,1,:,i+l])
                                Y[i,s,:,0] = tmp1
                                Y[i,s,:,1] = tmp2
                        tmp = 0
                        for k in range(self.K):
                            for i in range(min(self.N-1,n-1)):
                                tmp += z[k, n-i-1] * Y[i+1, index, k, :].conjugate()
                        Lambda[index, :] = abs(tmp+z[0,n]*Y[0,index,0,:].conjugate()+z[1,n]*Y[0,index,1,:].conjugate()) - abs(tmp) - 0.5*(abs(Y[0,index,0,:])**2+abs(Y[0,index,1,:])**2)
                        
                        state1 = state[0:2**(n+1)] // 2
                        index = state[0:2**(n+1)]
                        cost[index] = cost[state1] + Lambda[state1, state[index%2]]
                        path[index, n] = state1
                        alpha[index,:,:] = alpha_temp[state1, state[index%2], :,:]      
                else:
                    alpha_temp[:, 0, 0, 0] = operator[0] * alpha[:, 0, 0]
                    alpha_temp[:, 1, 0, 0] = operator[1] * alpha[:, 0, 0]
                    if n == 1:
                        alpha_temp[:, 0, 1, 0] = operator[0] * complex(0, 1)
                        alpha_temp[:, 1, 1, 0] = operator[1] * complex(0, 1)
                    else:
                        alpha_temp[:, 0, 1, 0] = operator[0] * alpha[:, 0, 1]
                        alpha_temp[:, 1, 1, 0] = operator[1] * alpha[:, 0, 1]
                    alpha_temp[:, :, :, 1:] = np.stack([alpha[:, :, :-1]]*2, axis=1)
                    for i in range(self.N):
                        for s in range(self.S):
                            tmp1 = 0
                            tmp2 = 0
                            for l in range(self.L+1):
                                tmp1 += np.matmul(F[:,:,l].T,alpha_temp[s,0,:,i+l])
                                tmp2 += np.matmul(F[:,:,l].T,alpha_temp[s,1,:,i+l])
                            Y[i,s,:,0] = tmp1
                            Y[i,s,:,1] = tmp2
                    tmp = 0
                    for k in range(self.K):
                        for i in range(min(self.N-1,n-1)):
                            tmp += z[k, n-i-1] * Y[i+1, :, k, :].conjugate()
                    Lambda[:, :] = abs(tmp+z[0,n]*Y[0,:,0,:].conjugate()+z[1,n]*Y[0,:,1,:].conjugate()) - abs(tmp) - 0.5*(abs(Y[0,:,0,:])**2+abs(Y[0,:,1,:])**2)
                    
                    state1 = state // 2
                    state2 = state1 + self.S//2
                    cost1 = cost[state1] + Lambda[state1, state%2]
                    cost2 = cost[state2] + Lambda[state2, state%2]
                    index = cost1 < cost2
                    cost = cost1
                    cost[np.where(index)] = cost2[index]
                    path[:, n] = state1
                    path[:, n][np.where(index)] = state2[index]
                    alpha[:,:,:] = alpha_temp[state1, state%2, :,:]
                    alpha[:,:,:][np.where(index)] = alpha_temp[state2, state%2, :,:][index]
        else:
            raise NotImplementedError
      
        max_id = np.argmax(cost)
        decode_seq = self.Decode(max_id, path)
        return decode_seq
    

class RCReciver(NonCoherentReceiver):
    def __init__(self, *args):
        super(RCReciver, self).__init__(*args)

    def WhitenedMatchedFilter(self, x):
        if self.K == 2:
            F = np.zeros([2,2,2])
            F[:,:,0] = np.array([0.11897,-0.093555,0.17232,-0.11299]).reshape(2,2)
            F[:,:,1] = np.array([0.69347,0,0.12821,0.15584]).reshape(2,2)   
            WF11 = np.array([1.8718,-1.3572])
            WF12 = np.array([1.1237])
            WF21 = np.array([-1.53995,-2.06987])
            WF22 = np.array([8.32966,1.42897])
            WF0 = np.array([1.2981,-0.57444,0.03218])
            z0 = signal.lfilter(WF11,WF0,x[0,...])+signal.lfilter(WF12,WF0,x[1,...])
            z1 = signal.lfilter(WF21,WF0,x[0,...])+signal.lfilter(WF22,WF0,x[1,...])
            z = np.zeros([2,self.s_len],dtype = complex)
            z[:,0:self.s_len-1] = np.array([z0[1:self.s_len], z1[1:self.s_len]])  
        elif self.K == 3:
            F = np.zeros([3,3,2])
            F[:,:,0] = np.array([0.1532,0.20435,0.0053,-0.0302,-0.019193,0.0028,-0.1278995,-0.079034,0.0043]).reshape([3,3])
            F[:,:,1] = np.array([0.6105,0.0379,0,0.2923,0.1778,0,0,0,0]).reshape([3,3])
            F[:,:,0] = F[:,:,0].T
            F[:,:,1] = F[:,:,1].T
            WF11 = np.array([-38.767,-7.03616])
            WF12 = np.array([63.719,11.57714])
            WF13 = np.array([18.3094,3.4453])
            WF21 = np.array([8.26484,65.803])
            WF22 = np.array([-133.138,-67.7852])
            WF23 = np.array([-2200.548,711.0978])
            WF31 = np.array([0,42.3998,-34.172])
            WF32 = np.array([0,8.12476,29.8682])
            WF33 = np.array([-4943.14,2183.314,-163.833])
            WF0 = np.array([-21.25,3.325,1.305])
            z0 = signal.lfilter(WF11,WF0,x[0,...])+signal.lfilter(WF12,WF0,x[1,...])+signal.lfilter(WF13,WF0,x[2,...])
            z1 = signal.lfilter(WF21,WF0,x[0,...])+signal.lfilter(WF22,WF0,x[1,...])+signal.lfilter(WF23,WF0,x[2,...])
            z2 = signal.lfilter(WF31,WF0,x[0,...])+signal.lfilter(WF32,WF0,x[1,...])+signal.lfilter(WF33,WF0,x[2,...])
            z = np.zeros([3,self.s_len],dtype = complex)
            z[0,0:self.s_len-1] = z0[1:self.s_len]
            z[1,0:self.s_len-1] = z1[1:self.s_len]
            z[2,0:self.s_len-2] = z2[2:self.s_len]
        else:
            raise NotImplementedError
        
        return z, F

    def ViterbiDemodulation(self):
        x = self.PulseMatchedFilter()
        z, F = self.WhitenedMatchedFilter(x)
        path = np.zeros([self.S, self.s_len])
        depth = np.log2(self.S) // 2
        cost = np.zeros(self.S)
        Lambda = np.zeros([self.S,4])
        operator = {0:[-1,-1],1:[1,-1],2:[-1,1],3:[1,1]}
        state = np.arange(self.S)

        if self.K == 2:
            Y = np.zeros([self.S, self.N, 2], dtype=complex)
            alpha = np.zeros([self.S, 2, self.N+self.L], dtype = complex)
            alpha_temp = np.zeros([self.S, 4, 2, self.N+self.L],dtype = complex)
            b1 = np.zeros([2, self.S], dtype=complex)
            b2 = np.zeros([2, self.S, 4], dtype=complex)
            for n in range(self.s_len):
                if n < depth:
                    if n == 0:
                        for i in range(4):
                            b2[0,0,i] = np.exp(1j*self.h*np.pi*operator[i][0])
                            b2[1,0,i] = np.exp(2j*self.h*np.pi*operator[i][1])
                            alpha_temp[0,i,0,0] = b2[0,0,i]*b2[1,0,i] 
                            alpha_temp[0,i,1,0] = b2[0,0,i]+b2[1,0,i]
                            Y[0, 0, :] = np.matmul(F[:,:,0].T,alpha_temp[0,i,:,0])
                            Lambda[0,i] = abs(z[0,0]*Y[0, 0, 0].conjugate()+z[1,0]*Y[0, 0, 1].conjugate())-0.5*np.sum(abs(Y[0, 0, :]*Y[0, 0, :].conjugate()))
                        
                        cost[0:4] = Lambda[0,0:4]
                        path[0:4,n] = 0
                        alpha[0:4,:,:] = alpha_temp[0,0:4,:,:]
                        b1[:,0:4] = b2[:,0,0:4]
                    else:
                        index = state[0:4**n]
                        for i in range(4):
                            b2[0, index, i] = b1[0, index] * np.exp(1j*self.h*np.pi*operator[i][0])
                            b2[1, index, i] = b1[1, index] * np.exp(2j*self.h*np.pi*operator[i][1])
                            alpha_temp[index,i,0,0] = b2[0,index,i]*b2[1,index,i]
                            alpha_temp[index,i,1,0] = b2[0,index,i]*b1[1,index]+b2[1,index,i]*b1[0,index]
                            alpha_temp[index,i,:,1:self.N+self.L] = alpha[index,:,0:self.N+self.L-1]
                            for j in range(self.N):
                                for s in range(4**n):
                                    Y[s, j, :] = np.matmul(F[:,:,0].T,alpha_temp[s,i,:,j]) + np.matmul(F[:,:,1].T,alpha_temp[s,i,:,j+1])
                            tmp = 0
                            for k in range(2):
                                for j in range(min(self.N-1, n-1)):
                                    tmp += z[k,n-j-1]*(Y[index, j+1, k].conjugate())
                            Lambda[index, i] = abs(tmp+z[0,n]*(Y[index,0,0].conjugate())+z[1,n]*(Y[index,0,1].conjugate()))-0.5*abs(tmp)-0.5*np.sum(abs(Y[index,0,:]*Y[index,0,:].conjugate()), axis=-1)

                        index = state[0:4**(n+1)]
                        state1 = index // 4
                        cost[index] = cost[state1] + Lambda[state1, index%4]
                        path[index, n] = state1
                        alpha[index,:,:] = alpha_temp[state1, index%4, :,:] 
                        b1[:, index] = b2[:, state1, index%4]
                else:
                    for i in range(4):
                        b2[0, :, i] = b1[0, :] * np.exp(1j*self.h*np.pi*operator[i][0])
                        b2[1, :, i] = b1[1, :] * np.exp(2j*self.h*np.pi*operator[i][1])
                        alpha_temp[:,i,0,0] = b2[0,:,i]*b2[1,:,i]
                        alpha_temp[:,i,1,0] = b2[0,:,i]*b1[1,:]+b2[1,:,i]*b1[0,:]
                        alpha_temp[:,i,:,1:self.N+self.L] = alpha[:,:,0:self.N+self.L-1]
                        for j in range(self.N):
                            for s in range(self.S):
                                Y[s, j, :] = np.matmul(F[:,:,0].T,alpha_temp[s,i,:,j]) + np.matmul(F[:,:,1].T,alpha_temp[s,i,:,j+1])
                        tmp = 0
                        for k in range(2):
                            for j in range(min(self.N-1, n-1)):
                                # n-dim arrays are shits, which takes me hours to debug
                                tmp += z[k,n-j-1]*(Y[:, j+1, k].conjugate())
                        Lambda[:, i] = abs(tmp+z[0,n]*(Y[:,0,0].conjugate())+z[1,n]*(Y[:,0,1].conjugate()))-0.5*abs(tmp)-0.5*np.sum(abs(Y[:,0,:]*Y[:,0,:].conjugate()), axis=-1)
                       
                    state1 = state // 4
                    state2 = state1 + self.S//4
                    state3 = state1 + 2*self.S//4
                    state4 = state1 + 3*self.S//4
                    
                    cost1 = cost[state1] + Lambda[state1, state%4]
                    cost2 = cost[state2] + Lambda[state2, state%4]
                    cost3 = cost[state3] + Lambda[state3, state%4]
                    cost4 = cost[state4] + Lambda[state4, state%4]
                    
                    for s in range(self.S):
                        costtot = np.array([cost1[s], cost2[s], cost3[s], cost4[s]])
                        statetot = np.array([state1[s], state2[s], state3[s], state4[s]])
                        max_id = np.argmax(costtot)
                        cost[s] = costtot[max_id]
                        path[s,n] = statetot[max_id]
                        alpha[s,:,:] = alpha_temp[statetot[max_id],s%4,:,:]
                        b1[:,s] = b2[:,statetot[max_id],s%4]
        elif self.K == 3:
            alpha = np.zeros([self.S,3,self.N+self.L],dtype = complex)
            alpha_temp = np.zeros([self.S,4,3,self.N+self.L],dtype = complex)
            Y = np.zeros([self.S,self.N,3],dtype = complex)
            b1 = np.zeros([2,2,self.S],dtype = complex)
            b2 = np.zeros([2,2,self.S, 4], dtype=complex)
            for n in range(self.s_len):
                if n < depth:
                    if n == 0:
                        for i in range(4):
                            b2[0,0,0,i] = np.exp(1j*self.h*np.pi*operator[i][0])
                            b2[1,0,0,i] = np.exp(2j*self.h*np.pi*operator[i][1])
                            alpha_temp[0,i,0,0] = b2[0,0,0,i]*b2[1,0,0,i] 
                            alpha_temp[0,i,1,0] = b2[0,0,0,i]+b2[1,0,0,i]
                            alpha_temp[0,i,2,0] = b2[0,0,0,i]+b2[1,0,0,i]
                            Y[0, 0, :] = np.matmul(F[:,:,0].T,alpha_temp[0,i,:,0])
                            Lambda[0,i] = abs(z[0,0]*Y[0, 0, 0].conjugate()+z[1,0]*Y[0, 0, 1].conjugate()+z[2,0]*Y[0, 0, 2].conjugate())-0.5*np.sum(abs(Y[0, 0, :]*Y[0, 0, :].conjugate()))
                        
                        cost[0:4] = Lambda[0,0:4]
                        path[0:4,n] = 0
                        alpha[0:4,:,:] = alpha_temp[0,0:4,:,:]
                        b1[:,:,0:4] = b2[:,:,0,0:4]
                    else:
                        index = state[0:4**n]
                        for i in range(4):
                            b2[0, 0, index, i] = b1[0, 0, index] * np.exp(1j*self.h*np.pi*operator[i][0])
                            b2[0, 1, index, i] = b1[0, 0, index]
                            b2[1, 0, index, i] = b1[1, 0, index] * np.exp(2j*self.h*np.pi*operator[i][1]), 
                            b2[1, 1, index, i] = b1[1, 0, index]
                            alpha_temp[index,i,0,0] = b2[0,0,index,i]*b2[1,0,index,i]
                            alpha_temp[index,i,1,0] = b2[0,0,index,i]*b1[1,0,index]+b2[1,0,index,i]*b1[0,0,index]
                            if n == 2:
                                alpha_temp[index,i,2,0] = b2[0,0,index,i]+b2[1,0,index,i]
                            else:
                                alpha_temp[index,i,2,0] = b2[0,0,index,i]*b1[1,1,index]+b2[1,0,index,i]*b1[0,1,index]
                            alpha_temp[index,i,:,1:self.N+self.L] = alpha[index,:,0:self.N+self.L-1]
                            for j in range(self.N):
                                for s in range(4**n):
                                    Y[s, j, :] = np.matmul(F[:,:,0].T,alpha_temp[s,i,:,j]) + np.matmul(F[:,:,1].T,alpha_temp[s,i,:,j+1])
                            tmp = 0
                            for k in range(self.K):
                                for j in range(min(self.N-1, n-1)):
                                    tmp += z[k,n-j-1]*(Y[index, j+1, k].conjugate())
                            Lambda[index, i] = abs(tmp+z[0,n]*(Y[index,0,0].conjugate())+z[1,n]*(Y[index,0,1].conjugate())+z[2,n]*(Y[index,0,2].conjugate()))-0.5*abs(tmp)-0.5*np.sum(abs(Y[index,0,:]*Y[index,0,:].conjugate()), axis=-1)

                        index = state[0:4**(n+1)]
                        state1 = index // 4
                        cost[index] = cost[state1] + Lambda[state1, index%4]
                        path[index, n] = state1
                        alpha[index,:,:] = alpha_temp[state1, index%4, :,:] 
                        b1[:, :, index] = b2[:, :, state1, index%4]
                else:
                    for i in range(4):
                        b2[0, 0, :, i] = b1[0, 0, :] * np.exp(1j*self.h*np.pi*operator[i][0])
                        b2[0, 1, :, i] = b1[0, 0, :]
                        b2[1, 0, :, i] = b1[1, 0, :] * np.exp(2j*self.h*np.pi*operator[i][1])
                        b2[1, 1, :, i] = b1[1, 0, :]
                        alpha_temp[:,i,0,0] = b2[0,0,:,i]*b2[1,0,:,i]
                        alpha_temp[:,i,1,0] = b2[0,0,:,i]*b1[1,0,:]+b2[1,0,:,i]*b1[0,0,:]
                        if n == 2:
                            alpha_temp[:,i,2,0] = b2[0,0,:,i]+b2[1,0,:,i]
                        else:
                            alpha_temp[:,i,2,0] = b2[0,0,:,i]*b1[1,1,:]+b2[1,0,:,i]*b1[0,1,:]
                        alpha_temp[:,i,:,1:self.N+self.L] = alpha[:,:,0:self.N+self.L-1]
                        for j in range(self.N):
                            for s in range(self.S):
                                Y[s, j, :] = np.matmul(F[:,:,0].T,alpha_temp[s,i,:,j]) + np.matmul(F[:,:,1].T,alpha_temp[s,i,:,j+1])
                        tmp = 0
                        for k in range(self.K):
                            for j in range(min(self.N-1, n-1)):
                                # n-dim arrays are shits, which takes me hours to debug
                                tmp += z[k,n-j-1]*(Y[:, j+1, k].conjugate())
                        Lambda[:, i] = abs(tmp+z[0,n]*(Y[:,0,0].conjugate())+z[1,n]*(Y[:,0,1].conjugate())+z[2,n]*(Y[:,0,2].conjugate()))-0.5*abs(tmp)-0.5*np.sum(abs(Y[:,0,:]*Y[:,0,:].conjugate()), axis=-1)
                       
                    state1 = state // 4
                    state2 = state1 + self.S//4
                    state3 = state1 + 2*self.S//4
                    state4 = state1 + 3*self.S//4
                    
                    cost1 = cost[state1] + Lambda[state1, state%4]
                    cost2 = cost[state2] + Lambda[state2, state%4]
                    cost3 = cost[state3] + Lambda[state3, state%4]
                    cost4 = cost[state4] + Lambda[state4, state%4]
                    
                    for s in range(self.S):
                        costtot = np.array([cost1[s], cost2[s], cost3[s], cost4[s]])
                        statetot = np.array([state1[s], state2[s], state3[s], state4[s]])
                        max_id = np.argmax(costtot)
                        cost[s] = costtot[max_id]
                        path[s,n] = statetot[max_id]
                        alpha[s,:,:] = alpha_temp[statetot[max_id],s%4,:,:]
                        b1[:,:,s] = b2[:,:,statetot[max_id],s%4]

        else:
            raise NotImplementedError
        max_id = np.argmax(cost)
        decode_seq = self.Decode(max_id, path)
        return decode_seq

