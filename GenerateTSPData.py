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
import os
from skimage.transform import resize
np.random.seed(1337)  # for reproducibility
import numpy as np
import sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
def plot_to_vec(plotname):

    img = plt.imread(plotname)
    img_r = resize(img, (img.shape[0] // 4, img.shape[1] // 4),
                       anti_aliasing=True)
    arr = np.array(img_r)
    # record the original shape
    shape = arr.shape
    # print(shape)
    arr.reshape(-1,120,160,1)

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
    
def generate_tsp_instances_data():
    #Clear data directory (For use in debugging)
    # pfiles = os.listdir('MyData/NGAplots')
    # for f in pfiles:
    #     fname='MyData/NGAplots/{}'.format(f)
    #     os.remove(fname)
            
        
    optimals = []

    for i in range(0,1000,50):
        for nc in range(i,i+50,2):
            cities=(int(i/10)+10)
            print("\n\n\n\n\ncities=",cities,'\n\n\n\n')
            print(nc)
            #Generate random x and y values for cities
            x_values = np.random.rand(cities)*100
            y_values = np.random.rand(cities)*100

            #Write city data to tsp file
            name='test{}'.format(str(nc))
            filename='MyData/coords/instance{}.tsp'.format(str(nc))
            fp=open(filename, 'w+')
            concorde.write_tsp_file(fp, x_values, y_values, "EUC_2D", name)
            fp.close()
        
            

            name='test{}'.format(str(nc))
            fname='MyData/coords/instance{}.tsp'.format(str(nc))
            with HiddenPrints():
                solver = concorde.TSPSolver.from_tspfile(fname)        
                solution=solver.solve(verbose=False)
            optimals.append(solution.optimal_value)
            
            plotname = str('MyData/NGAplots/plot{}.png'.format(str(nc)))
            plt.figure()
            plt.axis('off')
            ptx=[]
            pty=[]
           # print("Tour: _____ "+str(solution.tour))
            for c in solution.tour:
               # print(i)
                ptx.append(x_values[c])
                pty.append(y_values[c])
                
            ptx.append(x_values[0])
            pty.append(y_values[0])
            
            
            plt.scatter(ptx,pty)
            plt.plot(ptx, pty)
                
            plt.savefig(plotname)  
            #plt.show()
            plt.close()
            
            #print("Optimal value from solution: ", solution.optimal_value)
            pts=[]
            for g in range(len(x_values)):
                pts.append((x_values[g],y_values[g]))
        
            plotname = str('MyData/NGAplots/plot{}.png'.format(str(nc+1)))
            plt.figure()
            plt.axis('off')
            ptx=[]
            pty=[]
           # print("Tour: _____ "+str(solution.tour))
            
            
            # Run a round of random swaps
            idx = range(cities)
            r = random.sample(idx, random.randint(5,cities))
            # Randomly rearrange sample of indices
            copy = r.copy()
            random.shuffle(copy)
            pi_mod=solution.tour.copy()
            for j in range(len(r)):
                idx=int(r[j])
                pi_mod[idx] = solution.tour[copy[j]]
    
            for q in pi_mod:
               # print(i)
                ptx.append(x_values[q])
                pty.append(y_values[q])
                
            ptx.append(x_values[0])
            pty.append(y_values[0])

            plt.scatter(ptx,pty)
            plt.plot(ptx, pty)
                
            plt.savefig(plotname)  
            #plt.show()
            plt.close()
           
    return()
        

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

def mk_matrix(coord, dist=distL2):
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
            
        #plt.show()
        plt.savefig(plotfilename,)  
        plt.close()
       
generate_tsp_instances_data()



