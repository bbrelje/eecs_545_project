import numpy as np 
import matplotlib.pyplot as plt
from smt.surrogate_models import KPLS, KRG, RMTC, RMTB, KPLSK, RBF, LS, QP, IDW, MGP
from smt.applications import MOE 

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
print('Data points: '+str(counter)+' Skipped: '+str(skipcount))

# randomly split the data set into n batches and do CV error computations
np.random.seed(10)
np.random.shuffle(a)
n_split = 10

err = np.zeros((n_split, 7))
arrays = np.array_split(a, n_split, axis=0)

for i_array, arr in enumerate(arrays):
    # assemble the test and training set using holdout method
    test_set = arr
    training_set = np.zeros((0,6))
    for j in range(n_split):
        if j != i_array:
            training_set = np.vstack([training_set, arrays[j]])
    xt = training_set[:,0:3]
    yt = training_set[:,3]

    # train the model 

    # define a RMTS spline interpolant
    # TODO replace with a different SMT surrogate
    limits = np.array([[0.2,0.8],[0.05,1.0],[0.0,3.5]])
    sm = RMTB(print_global=False, order=3, xlimits=limits, nonlinear_maxiter=100)
    # sm1 = KRG(hyper_opt='TNC', corr='abs_exp')
    # sm2 = KPLS(n_comp=3, corr='abs_exp', hyper_opt='TNC')
    sm3 = KPLSK(print_global=False, n_comp=3, theta0=np.ones(3), corr='squar_exp')
    sm4 = QP(print_global=False, )
    sm5 = LS(print_global=False, )
    sm1 = KPLS(print_global=False, n_comp=3, theta0=np.ones(3), corr='abs_exp')
    sm2 = KRG(print_global=False, theta0=np.ones(3), corr='abs_exp')
    sm6 = MOE(smooth_recombination=False, n_clusters=2)
    experts_list = dict()
    experts_list['KRG'] = (KRG, {'theta0':np.ones(3), 'corr':'abs_exp'})
    experts_list['RBF'] = (RBF, dict())
    experts_list['KPLS'] = (KPLS, {'n_comp': 3, 'theta0':np.ones(3), 'corr':'abs_exp'})
    experts_list['KPLSK'] = (KPLSK, {'n_comp': 3, 'theta0':np.ones(3), 'corr':'squar_exp'})
    experts_list['QP'] = (QP, dict())
    experts_list['LS'] = (LS, dict())

    sm6.expert_types_manual = experts_list
    # sm = KRG(theta0=np.ones(3), corr='abs_exp')
    # sm = QP()
    # sm = LS()
    # sm = IDW(p=10.0)
    # sm = RBF()
    all_sms = [sm, sm1, sm2, sm3, sm4, sm5, sm6]
    for sm_ind, this_sm in enumerate(all_sms):
        this_sm.set_training_values(xt, yt)
        this_sm.train()
    
        # compute crossvalidation error for this batch
        err_sum = 0.0
        for j in range(test_set.shape[0]):
            predictions = np.zeros((1,3))
            mach = test_set[j,0]
            throttle = test_set[j,1]
            altitude = test_set[j,2]
            thrust_true = test_set[j,3]
            fuelburn_true = test_set[j,4]
            predictions = this_sm.predict_values(np.array([[mach, throttle, altitude]]))
            err_sum += np.abs((thrust_true - predictions[0,0]))
        avg_err = err_sum / test_set.shape[0]
        err[i_array, sm_ind] = avg_err

print('Avg abs error for thrust: '+str(np.mean(err, axis=0)))
# Avg abs error for thrust: [2.74772521e+01 1.29056146e+00 1.29779940e+00 5.91668556e-03
#  4.03213490e+02 1.16365692e+03 6.49398269e+01]