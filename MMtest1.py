#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:17:58 2020

@author: jimmyjacobson
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
import concorde.tsp as concorde
import glob
import os
from skimage.transform import resize, rescale
from dbn import SupervisedDBNClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.metrics.classification import accuracy_score
from sklearn import preprocessing
from dbn import SupervisedDBNRegression
# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy
import numpy as np
import skopt
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


def plot_to_vec(plotname):

    img = plt.imread(plotname)
    img_r = resize(img, (img.shape[0] // 4, img.shape[1] // 4),
                       anti_aliasing=True)
    arr = np.array(img_r)
    # record the original shape
    shape = arr.shape
    # print(shape)
    arr.reshape(-1,120,160,1)
    # # make a 1-dimensional view of arr
    # flat_arr = arr.ravel()
    # # convert it to a matrix
    # vector = np.array(flat_arr)
    
    # Plot rescaled image side by side
    # fig, axes = plt.subplots(nrows=2, ncols=1)
    # ax = axes.ravel()
    # ax[0].imshow(img, cmap='gray')
    # ax[0].set_title("Original image")
    
    # ax[1].imshow(img_r, cmap='gray')
    # ax[1].set_title("Rescaled image (aliasing)")
    # plt.tight_layout()
    # plt.close()
    
    return(arr, shape)

def plot_folder_to_vectorized_df(foldername):
    files = os.listdir(foldername)
    out = []
    for i in range(len(files)):
        p = foldername+'/'+str(files[i])
       # print(p)
        vec=plot_to_vec(p)
        #print(vec)
        out.append(vec[0])
    shape=vec[1]
    return(out,shape)
    
def generate_tsp_instances_data(cities, n):
    # #Clear data directory (For use in debugging)
    pfiles = os.listdir('MyData/plots')
    for f in pfiles:
        fname='MyData/plots/{}'.format(f)
        os.remove(fname)
    cfiles = os.listdir('MyData/coords')
    for f in cfiles:
        fname='MyData/coords/{}'.format(f)
        os.remove(fname)
    sfiles = os.listdir('MyData/sol_plots')
    for f in sfiles:
        fname='MyData/sol_plots/{}'.format(f)
        os.remove(fname)
    ifiles = os.listdir('MyData/SIplots')
    for f in ifiles:
        fname='MyData/SIplots/{}'.format(f)
        os.remove(fname)
            
        
    # Array to store vectorized img data
    pointdata = []
    solveddata = np.ndarray([n,76800])
    optimals = []
    num_SI_plots = 5
    rel_l_df = np.ndarray(num_SI_plots*n)
    #Generate data
    for nc in range(n):
            
        #Generate random x and y values for cities
        x_values = np.random.rand(cities)*100
        y_values = np.random.rand(cities)*100
        
        #Plot cities and write plot to file
        plt.figure()
        plt.axis('off')
        plt.scatter(x_values, y_values)
        plotname = str('MyData/plots/testplot{}'.format(str(nc)))
        plt.savefig(plotname)
        plt.close()
        
        #Write city data to tsp file
        name='test{}'.format(str(nc))
        filename='MyData/coords/test{}.tsp'.format(str(nc))
        fp=open(filename, 'w+')
        concorde.write_tsp_file(fp, x_values, y_values, "EUC_2D", name)
        fp.close()
    
    # # Read in plots as np arrays
    # for i in range(10):
        plotname = str('MyData/plots/testplot{}.png'.format(str(nc)))
        pointdata.append(plot_to_vec(plotname)[0])
        
    # # Solve TSP and plot
    # for i in range(10):
        name='test{}'.format(str(nc))
        fname='MyData/coords/test{}.tsp'.format(str(nc))
        solver = concorde.TSPSolver.from_tspfile(fname)        
        solution=solver.solve(verbose=False)
        optimals.append(solution.optimal_value)
        
        plotname = str('MyData/sol_plots/solplot{}.png'.format(str(nc)))
        plt.figure()
        plt.axis('off')
        ptx=[]
        pty=[]
       # print("Tour: _____ "+str(solution.tour))
        for i in solution.tour:
           # print(i)
            ptx.append(x_values[i])
            pty.append(y_values[i])
            
        ptx.append(x_values[0])
        pty.append(y_values[0])
        
        for i in range(cities):
            plt.scatter(ptx,pty)
            plt.plot(ptx, pty)
            
        plt.savefig(plotname)  
        plt.show()
        plt.close()
        
        #print("Optimal value from solution: ", solution.optimal_value)
        pts=[]
        for i in range(len(x_values)):
            pts.append((x_values[i],y_values[i]))
    
        dm = mk_matrix(pts,distL2)[1]
        iteration=nc
        #print('iteration: ', iteration)
        rel_l_df = single_instance_perms(x_values,y_values,20,num_SI_plots,solution.tour, dm,iteration,rel_l_df)
    
    return([pointdata,optimals,x_values,y_values,solution,rel_l_df])
    
def swap_random(seq):
    idx = range(len(seq))
    i1,i2 = random.sample(idx, 2)
    seq[i1], seq[i2] = seq[i2], seq[i1]
    return(seq)

def distL2(x1,y1,x2,y2):
    """Compute the L2-norm (Euclidean) distance between two points.

    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.

    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters"""
    xdiff = x2 - x1
    ydiff = y2 - y1
    return int(math.sqrt(xdiff*xdiff + ydiff*ydiff) + .5)

def length(tour, D):
    """Calculate the length of a tour according to distance matrix 'D'."""
    
    z = D[(tour[-1], tour[0])]    # edge from last to first city of the tour
    for i in range(1,len(tour)):
        z += D[tour[i], tour[i-1]]      # add length of edge from city i-1 to i
    return z

def mk_matrix(coord, dist):
    """Compute a distance matrix for a set of points.

    Uses function 'dist' to calculate distance between
    any two points.  Parameters:
    -coord -- list of tuples with coordinates of all points, [(x1,y1),...,(xn,yn)]
    -dist -- distance function
    """
    n = len(coord)
    D = {}      # dictionary to hold n times n matrix
    for i in range(n-1):
        for j in range(i+1,n):
            (x1,y1) = coord[i]
            (x2,y2) = coord[j]
            D[i,j] = dist(x1,y1,x2,y2)
            D[j,i] = D[i,j]
    return n,D

def plot_permutation(pi, plotfilename, x_values,y_values):
    #Takes a permutation, and points as a list of tuples
        plt.figure()
        plt.axis('off')
        ptx=[]
        pty=[]
        for i in pi:
            ptx.append(x_values[i])
            pty.append(y_values[i])
            
        ptx.append(x_values[0])
        pty.append(y_values[0])
        
        for i in range(len(pi)):
            plt.scatter(ptx,pty)
            plt.plot(ptx, pty)
            
        plt.show()
        plt.savefig(plotfilename)  
        plt.close()
       
    
def single_instance_perms(x_values,y_values, ncities, num, opt_perm, dm, iteration,rel_l_df):
    ## Generates num different permutations of pts given 
    ## ncities points as a list of tuples, plots them, and 
    ## saves them
    perm=opt_perm
    opt_value = length(perm,dm)
   # print("Optimal value from length: ", opt_value)
    for i in range(num*iteration,num*(iteration+1)):
        pnum=i
      #  print(pnum)
        plotname = str('MyData/SIplots/SIplot{}.png'.format(str(pnum)))
        plot_permutation(perm, plotname, x_values,y_values)
        l = length(perm,dm)
       # print("Postswap value from length: ", l)
        if l-opt_value != 0:
            optimal_ratio = (1/(l/opt_value))
        else:
            optimal_ratio = 1
        rel_l_df[pnum]=optimal_ratio
        perm=swap_random(perm)
    return(rel_l_df)


data = generate_tsp_instances_data(20, 100)
X = data[0]
#X = normalize(X, axis=1, norm='l2')
Y = data[1]
x_values=data[2]
y_values=data[3]
solution=data[4]

#print("Optimal value from solution: ", solution.optimal_value)
pts=[]
for i in range(len(x_values)):
    pts.append((x_values[i],y_values[i]))
    
dm = mk_matrix(pts,distL2)[1]

#si_Y = single_instance_perms(x_values,y_values,20,5,solution.tour, dm)


#si_X = np.array(si_X).reshape(-1,120,160,1)
#si_X = normalize(si_X, axis=1, norm='l2')
# Y_n = np.ndarray([10,76800])
# scaler = MinMaxScaler(feature_range=(0,1))


# for i in range(10):
#     plotname = str('MyData/sol_plots/solplot{}.png'.format(str(i)))
#     Y_n[i]= plot_to_vec(plotname)[0]
#     shape=plot_to_vec(plotname)[1]
    
# Y = normalize(Y_n, axis=1, norm='l2')
# arr2 = np.asarray(solveddata[4]).reshape(shape)
# img2 = plt.imshow(arr2)
# img2.show()


# #Training
# SPACE = [skopt.space.Real(0.0001, 0.01, name='_learning_rate', prior='log-uniform'),
#           skopt.space.Real(0.0001, 0.01, name='_learning_rate_rbm', prior='log-uniform'),
#           skopt.space.Integer(4, 8, name='_batch_size'),
#           skopt.space.Integer(10, 20, name='_hidden_layers_s')]
         
# @skopt.utils.use_named_args(SPACE)
# def objective(_learning_rate, _learning_rate_rbm, _batch_size, _hidden_layers_s):
#     X_train, X_test, Y_train, Y_test = train_test_split(si_X, si_Y, test_size=0.2, random_state=0)

#     min_max_scaler = MinMaxScaler()
#     X_train = min_max_scaler.fit_transform(X_train)
    
#     # Training
#     regressor = SupervisedDBNRegression(hidden_layers_structure=[_hidden_layers_s],
#                                         learning_rate_rbm=_learning_rate_rbm,
#                                         learning_rate=_learning_rate,
#                                         n_epochs_rbm=50,
#                                         n_iter_backprop=50,
#                                         batch_size=_batch_size,
#                                         activation_function='relu')
#     regressor.fit(X_train, Y_train)
    
#     # Test
#     X_test = min_max_scaler.transform(X_test)
#     Y_pred = regressor.predict(X_test)
#     for i in range(len(Y_pred)):
#         print("\nCorrect: \t"+str(Y_test[i]))
#         print("Predicted: \t"+str(Y_pred[i]))
#     print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))
#     return(1/mean_squared_error(Y_test, Y_pred))

batch_size = 64
epochs = 5

fashion_model = Sequential()
fashion_model.add(Conv2D(64, kernel_size=(10,10),activation='relu',input_shape=(120,160,4),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((10, 10),padding='same'))
fashion_model.add(Dropout(0.9))
#fashion_model.add(Dropout(0.5))
# fashion_model.add(Conv2D(64, (12), activation='relu',padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
# fashion_model.add(MaxPooling2D(pool_size=(20,20),padding='same'))
# fashion_model.add(Conv2D(128, (12), activation='relu',padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))                  
# fashion_model.add(MaxPooling2D(pool_size=(20,20),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='relu'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dropout(0.9))
fashion_model.add(Dense(1,activation='sigmoid'))

fashion_model.compile(loss=keras.losses.mean_squared_error, 
                      optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

si_X = plot_folder_to_vectorized_df('MyData/SIplots')[0]
si_Y=data[5]
l = len(si_X)
X_train = np.array(si_X[:int(.8*l)])
X_test = np.array(si_X[int(.8*l):])
Y_train = np.array(si_Y[:int(.8*l)])
Y_test = np.array(si_Y[int(.8*l):])


#X_train, X_test, Y_train, Y_test = train_test_split(si_X, si_Y, test_size=0.2, random_state=0)

#X_train, X_test, valid_X, valid_Y = train_test_split(X_train, Y_train, test_size=0.5, random_state=0)

# min_max_scaler = MinMaxScaler()
# X_train = min_max_scaler.fit_transform(X_train)
# X_test = min_max_scaler.transform(X_test)
# # 
#,validation_data=(X_valid, Y_valid)
fashion_train = fashion_model.fit(X_train, Y_train, batch_size=batch_size,epochs=epochs,verbose=1)

test_eval = fashion_model.predict(X_test, verbose=1)
for i in range(len(test_eval)):
    print("\nCorrect: \t"+str(Y_test[i]))
    print("Predicted: \t"+str(test_eval[i]))


# results = skopt.forest_minimize(objective, dimensions=SPACE,n_calls=10)






