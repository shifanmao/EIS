import matplotlib.pyplot as plt
import seaborn as sns
import RandomWalk as RandomWalk

nump = 10
for ip in range(nump):
    print('Particle #'+str(ip))
    
    # parameters
    dc = 1
    minv = 0.005
    tsteps = 1000

    rw = RandomWalk(dc=dc, minv=minv, tsteps = tsteps)
    rw.fill_walk()

#     # calculate and plot msd
#     tint, msd = rw.msd(avg='slide')
#     plt.scatter(tint,msd,s=5)

#     t = np.linspace(0,tsteps,tsteps)
#     y = 6*dc*t
#     plt.plot(t,y,linewidth=1.5)
    
    # calculate and plot velocity autocorrelation
    tint, vcorr = rw.velcorr(avg='slide')
    plt.plot(tint, vcorr/vcorr[1],linewidth = 0.5)

plt.xscale('linear')
plt.yscale('log')

t = tint
y = np.exp(-minv*t)
plt.plot(t,y,linewidth=1.5)
plt.show()
