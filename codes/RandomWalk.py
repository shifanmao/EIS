import numpy as np
from numpy import linalg as LA
from scipy.stats import norm

class RandomWalk():
    # this class simulates random walks
    # with diffusion constant dc

    def __init__(self, dc = 1, tsteps=1000, minv = 0):
        # constants
        self.d = dc
        self.tsteps = tsteps
        self.minv = minv

        # variables
        self.x = np.array([[0,0,0]])
        self.v = [norm.rvs(scale = (2*self.d)**(1/2), size = 3)]
        self.t = np.linspace(0, tsteps, tsteps)
    
    def fill_walk(self, dimension=3, pd='delta'):
        while len(self.x) <= self.tsteps:
            if pd=='delta':
                d = self.d
            brownian = norm.rvs(scale = np.sqrt(2*d), size = 3)
            if self.minv > 0:
                v = self.v[-1] + self.minv*(-self.v[-1] + brownian)
            else:
                v = brownian
            self.v=np.append(self.v, [v], axis=0)
            self.x=np.append(self.x, [self.x[-1]+v], axis=0)

    def msd(self, avg='window'):
        # calculate MSD
        tsteps = self.tsteps
        msd = np.array([0])
        tint = np.array([0])

        if avg=='window':
            for interv in range(1,tsteps):
                times1 = range(0,tsteps-interv,interv)
                times2 = range(interv,tsteps,interv)
                
                tint = np.append(tint, self.t[times2[0]] - self.t[times1[0]])
                
                sd = self.x[times2] - self.x[times1]
                sd = LA.norm(sd,axis=1)**2
                msd = np.append(msd,np.average(sd))
        elif avg=='slide':
            for interv in range(1,tsteps):
                times1 = range(0,tsteps-interv,1)
                times2 = range(interv,tsteps,1)

                tint = np.append(tint, self.t[times2[0]] - self.t[times1[0]])
                
                sd = self.x[times2] - self.x[times1]
                sd = LA.norm(sd,axis=1)**2
                msd = np.append(msd,np.average(sd))                
        return tint, msd
    
    def velcorr(self, avg='window'):
        # calculate velocity autocorrelation
        tsteps = self.tsteps
        vcorr = np.empty(())
        tint = np.empty(())
        
        if avg=='window':
            for interv in range(1,tsteps):
                times1 = range(0,tsteps-interv,interv)
                times2 = range(interv,tsteps,interv)
                tint = np.append(tint, self.t[times2[0]] - self.t[times1[0]])
                
                velcorr = np.einsum('ij,ij->i', self.v[times1], self.v[times2])
                vcorr = np.append(vcorr, np.average(velcorr))
        elif avg=='slide':
            for interv in range(1,tsteps):
                times1 = range(0,tsteps-interv,1)
                times2 = range(interv,tsteps,1)
                tint = np.append(tint, self.t[times2[0]] - self.t[times1[0]])
                
                velcorr = np.einsum('ij,ij->i', self.v[times1], self.v[times2])
                vcorr = np.append(vcorr, np.average(velcorr))
        return tint[2:], vcorr[2:]

#    def external_force(self,force=1):
