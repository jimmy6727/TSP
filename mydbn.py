import numpy as np
import re
import math
np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.preprocessing import normalize
from dbn import SupervisedDBNClassification
# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy
import numpy as np
import re
from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path
import time
import os

def distL2(x1,y1,x2,y2):
    """Compute the L2-norm (Euclidean) distance between two points.

    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.

    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters"""
    xdiff = x2 - x1
    ydiff = y2 - y1
    return int(math.sqrt(xdiff*xdiff + ydiff*ydiff) + .5)


def distL1(x1,y1,x2,y2):
    """Compute the L1-norm (Manhattan) distance between two points.

    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.

    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters"""
    return int(abs(x2-x1) + abs(y2-y1)+.5)


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

def read_tsplib(filename):
    "basic function for reading a TSP problem on the TSPLIB format"
    "NOTE: only works for 2D euclidean or manhattan distances"
    f = open(filename, 'r');

    line = f.readline()
    while line.find("EDGE_WEIGHT_TYPE") == -1:
        line = f.readline()

    if line.find("EUC_2D") != -1:
        dist = distL2
    elif line.find("MAN_2D") != -1:
        dist = distL1
    else:
        print ("cannot deal with non-euclidean or non-manhattan distances")
        raise Exception

    while line.find("NODE_COORD_SECTION") == -1:
        line = f.readline()

    xy_positions = []
    while 1:
        line = f.readline()
        if line.find("EOF") != -1: break
        (i,x,y) = line.split()
        x = float(x)
        y = float(y)
        xy_positions.append((x,y))

    n,D = mk_matrix(xy_positions, dist)
    return n, xy_positions, D

def read_tsp_data(tsp_name):
	tsp_name = tsp_name
	with open(tsp_name) as f:
		content = f.read().splitlines()
		cleaned = [x.lstrip() for x in content if x != ""]
		return cleaned
    
def get_dmax(a):
    n = len(a)
    cost_matrix = np.ndarray([n,n])
    for i in range(len(a)):
        l = a[i].split()
        cost_matrix[i]=l
    return(cost_matrix)

def cost_tour(fname):
        
    #dm_data_file = str(fname+"_d.txt")
    
    filename = "concorde/tests/data/{}.tsp".format(fname)
    solver = TSPSolver.from_tspfile(filename)
        # Problem Name: berlin52
        # Problem Type: TSP
        # 52 locations in Berlin (Groetschel)
        # Number of Nodes: 52
        # Rounded Euclidean Norm (CC_EUCLIDEAN)
        
    solution=solver.solve()
    #op = list(solution.tour)
    
    time.sleep(0.5)
    print(fname+"_d.txt")
    distance_mat = get_dmax(read_tsp_data(fname+"_d.txt"))
    print(distance_mat)
    normed = normalize(distance_mat, axis=1, norm='l2')
    # d = 0
    # for i in range(len(op)):
    #     ci = op[i]
    #     cj = op[(i+1)%26]
    #     add=distance_mat[ci][cj]
    #     d+=add
        
    vectorized = np.concatenate(normed)
    target = solution.optimal_value
    
    return [vectorized,target]

def build_data(maxlen, filelist):
    data = np.ndarray([ len(filelist),maxlen*maxlen])
    targets = []
    for instance in range(len(filelist)):
        row = cost_tour(filelist[instance])
        targets.append(row[1])
        if len(row[0]) == maxlen*maxlen:
            data[instance]=row[0]
        else: 
            buflen = (maxlen*maxlen-(len(row[0])))/2
            print(len(row[0]))
            print(buflen)
            modrow=np.pad(row[0], (math.floor(buflen),math.ceil(buflen)), 'constant', constant_values=(0,0))
            data[instance]=modrow
    targets_array = np.array(targets)
    return(data,targets_array)
      
#t = read_tsplib('concorde/tests/data/dj38.tsp')
      
d = build_data(48, ['att48','dantzig42'])
X, Y = d[0], d[1]




# Data scaling
#X = (X / 16).astype(np.float32)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training
#SPACE = [skopt.space.Real(0.01, 0.5, name='_learning_rate', prior='log-uniform'),
    #     skopt.space.Real(0.01, 0.5, name='_learning_rate_rbm', prior='log-uniform'),
     #    skopt.space.Real(0.01, 0.5, name='_dropout', prior='log-uniform'),
     #    skopt.space.Integer(4, 64, name='_batch_size')]
         
#@skopt.utils.use_named_args(SPACE)
def objective(X,Y, _learning_rate, _learning_rate_rbm, _batch_size, _dropout):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    classifier = SupervisedDBNClassification(hidden_layers_structure=[2304, 2304, 2304],
                                              learning_rate_rbm=_learning_rate_rbm,
                                              learning_rate=_learning_rate,
                                              n_epochs_rbm=50,
                                              n_iter_backprop=50,
                                              batch_size=_batch_size,
                                              activation_function='relu',
                                              dropout_p=_dropout)
    classifier.fit(X, Y)
    
    # Save the model
    classifier.save('model.pkl')
    
    # Restore it
    classifier = SupervisedDBNClassification.load('model.pkl')
    
    # Test
    Ypred = classifier.predict(X_test)
    print(Ypred)
    print("291 is correct. Predicted: "+str(Ypred))
    
    return((291-Ypred)/291)
     
#results = skopt.forest_minimize(func=objective, dimensions=SPACE)
objective(X,Y,0.05, 0.1,32,0.2)
