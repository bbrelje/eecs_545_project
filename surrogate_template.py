import numpy as np 
import matplotlib.pyplot as plt
from smt.surrogate_models import KPLS, KRG, RMTC, RMTB, KPLSK, RBF, LS, IDW

# load data
altdata = np.load(r'power_off/alt.npy')
throttledata = np.load(r'power_off/throttle.npy')
machdata = np.load(r'power_off/mach.npy')
thrustdata = np.load(r'power_off/thrust.npy')
fuelburndata = np.load(r'power_off/wf.npy')
tempdata = np.load(r'power_off/t4.npy')



# turn the grid of data into a flat vector of training points
# first three items are the independent vars
# last three are regressed variables
# skip if thrust = 0.0 (that means the analysis didn't converge - bad data point)

krigedata = []
counter = 0
skipcount =0 
for i in range(machdata.shape[0]):
    for j in range(throttledata.shape[1]):
        for k in range(altdata.shape[2]):
            thrustijk = thrustdata[i, j, k]
            if thrustijk > 0.0:
                krigedata.append(np.array([machdata[i,j,k].copy(), 
                                           throttledata[i,j,k].copy(), 
                                           altdata[i,j,k].copy()/10000,
                                           thrustijk.copy(), 
                                           fuelburndata[i,j,k].copy(), 
                                           tempdata[i,j,k].copy()]))
                counter += 1
            else:
                skipcount += 1

a = np.array(krigedata)
xt = a[:,0:3]
yt = a[:,3:]

print('Data points: '+str(counter)+' Skipped: '+str(skipcount))

# define a RMTS spline interpolant
# TODO replace with a different SMT surrogate
limits = np.array([[0.2,0.8],[0.05,1.0],[0.0,3.5]])
sm = RMTB(order=3, xlimits=limits, nonlinear_maxiter=100)
sm.set_training_values(xt, yt)
sm.train()

# plot a grid of values at a slice of throttle = 0.5

machs = np.linspace(0.2, 0.8, 25)
alts = np.linspace(0.0, 35000., 25)
machs, alts = np.meshgrid(machs, alts)
pred = np.zeros((25, 25, 3))
pred2 = np.zeros((25, 25, 3))

# altitude is scaled by 1 / 10000 everywhere to make the indepdent variables of O(1)
# this is necessary for certain methods like IDW
# TODO examine other slices (for example holding Mach = 0.5 and varying other two parameters)
for i in range(25):
    for j in range(25):
        pred[i,j,:] = sm.predict_values(np.array([[machs[i,j], 0.5, alts[i,j]/10000]]))
                            #                      mach        throttle altitude (scaled)
plt.figure()
plt.xlabel('Mach')
plt.ylabel('Altitude')
plt.title('Thrust (lb)')
plt.contourf(machs, alts, pred[:,:,0])
plt.colorbar()
plt.figure()
plt.xlabel('Mach')
plt.ylabel('Altitude')
plt.title('Fuel Burn (lb/s)')
plt.contourf(machs, alts, pred[:,:,1])
plt.colorbar()
plt.figure()
plt.xlabel('Mach')
plt.ylabel('Altitude')
plt.title('Turbine Inlet Temp (R)')
plt.contourf(machs, alts, pred[:,:,2])
plt.colorbar()
plt.show()