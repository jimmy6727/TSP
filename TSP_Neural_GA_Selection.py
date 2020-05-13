#import pickle 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import operator
import math

from skimage.transform import resize
#from TSPPlotModelTest import main as NeuralSelection
import time
import skopt
import os
import re
from keras.models import load_model
from PIL import Image


def plot_to_vec(plotname):
    
    img = Image.open(plotname).convert('L')
    arr = np.array(img)
    shape=arr.shape
    # print('\nData Type: %s' % arr.dtype)
    # print('Min: %.3f, Max: %.3f' % (arr.min(), arr.max()))
    # print(shape)
    img_r = resize(arr, output_shape=(120,160,1), anti_aliasing=True, preserve_range=True)
    img_r=((img_r-img_r.min()) / (img_r.max()-img_r.min())) * (254)+1
    img_r = img_r / 255
    #new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    # print('Data Type: %s' % img_r.dtype)
    # print('Min: %.3f, Max: %.3f' % (img_r.min(), img_r.max()))
    # print(str(img_r.shape))
        
    return(img_r, shape)

# Euclidean distance function
def L2dist(x1,y1,x2,y2):
    xdiff = x2 - x1
    ydiff = y2 - y1
    return int(math.sqrt(xdiff*xdiff + ydiff*ydiff) + .5)

def plot_folder_to_vectorized_df(foldername):
    files = os.listdir(foldername)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    outs = []
    #labels = []
    for i in range(len(files)):
        p = foldername+'/'+str(files[i])
        vec=plot_to_vec(p)
        #print(vec.shape)
        outs.append(vec[0])

    return(np.array(outs))

def mk_matrix(coord):
    """Compute a distance matrix for a set of points.

    Uses function 'dist' to calculate distance between
    any two points.  Parameters:
    -coord -- list of tuples with coordinates of all points, [(x1,y1),...,(xn,yn)]
    -dist -- distance function
    """
    n = len(coord)
    D = np.ndarray([n,n])     # array to hold n times n matrix
    for i in range(n-1):
        for j in range(i+1,n):
            (x1,y1) = coord[i]
            (x2,y2) = coord[j]
            D[i,j] = L2dist(x1,y1,x2,y2)
            D[j,i] = D[i,j]
    #Set distance from city to itself to zero
    np.fill_diagonal(D, 0)
    return D

def length(tour, D):
    """Calculate the length of a tour according to distance matrix 'D'."""
    
    z = D[(tour[-1], tour[0])]    # edge from last to first city of the tour
    for i in range(1,len(tour)):
        z += D[tour[i], tour[i-1]]      # add length of edge from city i-1 to i
    return z


def normalize(a, axis=(1,2)): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

def dm_normalize(a):
    mean = np.mean(a)
    std = np.sqrt(((a - mean)**2)).mean()
    return (a - mean) / std

class City:
    def __init__(self, index, x, y):
        self.index = index
        self.x = x
        self.y = y
        
    def index(self):
        return(str(self.index))
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def to_tuple(self):
        return((self.x,self.y))
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
    
class Fitness:
    
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        #print(self.route)        if self.fitness == 0            #p = predict_route_prob(self.route)
            p=1
            #print("Predicted route probability: ", str(p))
            self.fitness = (float(p) / float(self.routeDistance()))
            #self.fitness = p*self.fitness
            return self.fitness
    
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    res = sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
    #print(res)
    return res

def neuralRankRoutes(population, NeuralSelectionModel, maxgen):
    global plot_counter
    pfiles = os.listdir('MyData/NeuralGAPlots')
    for f in pfiles:
        fname='MyData/NeuralGAPlots/{}'.format(f)
        os.remove(fname)
        
    fitnessResults = {}
    pred = []
    for num in range(len(population)):
        i = population[num]
        plt.figure()
        plt.axis('off')
        t_x = [b.to_tuple()[0] for b in i]
        t_y = [c.to_tuple()[1] for c in i]
        t_x.append(i[0].to_tuple()[0])
        t_y.append(i[0].to_tuple()[1])
        plt.scatter(t_x,t_y)
        plt.plot(t_x,t_y)
        plotname = str('MyData/NeuralGAPlots/testplot{}.png'.format(str(num)))
        plt.savefig(plotname)
        plt.close()
        plot_to_vec(plotname)
        
    #plt.close('all')
        
    v = plot_folder_to_vectorized_df('MyData/NeuralGAPlots')
    #v = normalize(v)
    y = v[0]
    print("Sample Test Batch Stats: ", v.shape, round(v.mean(),3), round(v.std(),3), round(v.min(), 3),round(v.max(),3))
    pred = NeuralSelectionModel.predict(v)

    for i in range(0,len(population)):
        fitnessResults[i] = round(float(pred[i]),5)
    
    inorder = sorted([fitnessResults[i] for i in fitnessResults.keys()], reverse=True)
    global plots
    global fig
    x = v[next(key for key, value in fitnessResults.items() if value == inorder[0])]  
    x = x.reshape(120,160)  # this is a Numpy array with shape (1, 3, 150, 150)
    # sub = fig.add_subplot()
    # sub.imshow(x*255)
    # sub.set_title(str(round(float(inorder[0]),5)))
    plots.append(x)
    plot_titles.append(str(round(float(inorder[0]),5)))
    
    # show plots in debugging
        # f, axarr = plt.subplots(4,4)
        # for i in range(16):
        #     for i in range(4):
        #         for j in range(4):
        #             x = v[next(key for key, value in fitnessResults.items() if value == inorder[(4*i)+j])]  
        #             x = x.reshape(120,160)  # this is a Numpy array with shape (1, 3, 150, 150)
        #             axarr[i,j].imshow(x*255)
        #             axarr[i,j].set_title(str(round(float(inorder[(4*i)+j]),5)))
        # f.suptitle("Sample of Neural Selection Predictions")
        # f.show()
    
    res = sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
    #print(res)
    #plot_counter += 1
    return res

def neuralSelection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    print(df.head(3))
    print('...')

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
            
    return selectionResults

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
        
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    #print('Parent 1: ',parent1)
    #print('Parent 2: ', parent2)
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    #print(child)
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

# This is the control (not neural) mutate function
def mutate(individual, mutationRate,d_matrix):
    #print("Mutate call")
    nC=len(individual)
    idx = range(nC)
    mr = math.ceil(nC*mutationRate)
    # Take a random sample of indices according to the mutation rate
    r = random.sample(idx, mr)
    # Randomly rearrange sample of indices
    copy = r.copy()
    random.shuffle(copy)
    indices = [i.index for i in individual]
    pi_mod=indices.copy()
    #print(pi_mod)
    #print(indices)
    for j in range(len(r)):
        idx=int(r[j])
        pi_mod[idx] = indices[copy[j]]
    
    if length(pi_mod,d_matrix) < length(indices,d_matrix):
        return [individual[i] for i in pi_mod]
    else:
        return individual
    
    
# This is the neural mutate function  - unused in project
# def mutate(model,normalized_dm,individual, mutationRate, nC):
#     # Run a round of random mutations
#     idx = range(nC)
#     mr = math.ceil(nC*mutationRate)
#     # Take a random sample of indices according to the mutation rate
#     r = random.sample(idx, mr)
#     # Randomly rearrange sample of indices
#     copy = r.copy()
#     random.shuffle(copy)
#     pi_mod=individual.copy()
#     for j in range(len(r)):
#         idx=int(r[j])
#         pi_mod[idx] = individual[copy[j]]
        
#     # Convert individuals/tours to adjacency matrices for input into model
#     A = Permutation([i.index for i in individual])
#     #print("Tour A: \t", A.array_form)
#     A_arr=np.array(A.get_adjacency_matrix()).astype(np.float64)
#     #print(A_arr.shape)
#     B = Permutation([i.index for i in pi_mod])
#     #print("Tour B: \t", B.array_form)
#     B_arr=np.array(B.get_adjacency_matrix()).astype(np.float64)

#     predicted = model.predict([A_arr.reshape((1,nC,nC)),B_arr.reshape((1,nC,nC)),normalized_dm.reshape((1,nC,nC))])
   
#     #Investigate individual model predictions
#     # print("\nTour A: ", A.array_form)
#     # print("Tour B: ", B.array_form)
#     # print("Tour A length: \t", length(A.array_form, d_matrix))
#     # print("Tour B length: \t", length(B.array_form, d_matrix))
#     # print("Predicted value from model: \t", predicted)
#     if predicted > 0.2:
#         return pi_mod
#     else:
#         return individual

def mutatePopulation(population, mutationRate, normalized_dm):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate, normalized_dm)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(model, normalized_dm,currentGen, eliteSize, mutationRate, gen, maxgen):
    if gen <= 50 and (gen%3 == 0 or gen == maxgen-1 or gen ==0):
            popRanked = neuralRankRoutes(currentGen, NeuralSelectionModel = model, maxgen=True)
            selectionResults=neuralSelection(popRanked, eliteSize)
    elif 80 < gen <= 1000 and (gen%100 == 0 or gen == maxgen-1 or gen ==0):
            popRanked = neuralRankRoutes(currentGen, NeuralSelectionModel = model, maxgen=True)
            selectionResults=neuralSelection(popRanked, eliteSize)
    else:
        popRanked = rankRoutes(currentGen)
        selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate, normalized_dm)
    return nextGeneration

def geneticAlgorithm(model, normalized_dm, popSize, eliteSize, mutationRate, generations,cityList, nC):
    time_start=time.time()
    population=cityList
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    progress = []
    best=9999999999
    progress.append(1 / rankRoutes(pop)[0][1])
    for i in range(0, generations):
        pop = nextGeneration(model,normalized_dm,pop, eliteSize, mutationRate, gen = i, maxgen=generations)
        progress.append(1 / rankRoutes(pop)[0][1])
        n_best = 1 / rankRoutes(pop)[0][1]
        if n_best < best:
            best=n_best
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    print("Shortest route found: ", best)
    #line2, = plt.plot(progress, label = "Neural TI Agent Assisted")
    #legend1 = plt.legend(handles=[line2], loc='lower right')
    #plt.gcs.add_artist(legend1)
    #plt.ylabel('Distance of best route found')
    #plt.xlabel('Generation')
   # plt.show()
    fd = (1 / rankRoutes(pop)[0][1])
    time_finish=time.time()
    time_e = time_finish-time_start
    print("time elapsed: ", time_e)
#    times.update(time_e = [popSize, eliteSize, mutationRate])
    #return line2, bestRoute, time_e, best
    return bestRoute, time_e, best

# SPACE = [skopt.space.Integer(40, 100, name='popSize'),
#       skopt.space.Real(0.1, 0.9, name='mutationRate'),
#       #skopt.space.Categorical([4,8,16,32,64], name='_conv_layer_size'),
#       #skopt.space.Categorical([4,8,16,32,64, 128], name='_dense_layer_size'),
#       skopt.space.Integer(5, 40, name='eliteSize'),
#       skopt.space.Integer(5, 10, name='generations')]

#@skopt.utils.use_named_args(SPACE)
nCities = 30
cityList = []

for i in range(0,nCities):
    cityList.append(City(index=i,x=int(random.random()*100), y=int(random.random()*100)))

def main(mutationRate, popSize,eliteSize,generations,cityList, nCities):
    
    # Get cities and normalized distance matrix for input into model
    city_pts = [i.to_tuple() for i in cityList]
    d_matrix = mk_matrix(city_pts)
    normalized_dm = dm_normalize(d_matrix)
    
    model=load_model('TSPplotmodel1.h5')
    
    l=geneticAlgorithm(model,normalized_dm,popSize=popSize, eliteSize=eliteSize, mutationRate=mutationRate, generations=generations, cityList=cityList, nC=nCities)
    
    return(l)


# times = {}
# # results = skopt.forest_minimize(main, dimensions=SPACE,
# #                                     n_calls=30,
# #                                     base_estimator='RF',
# #                                     acq_func='PIps',
# #                                     verbose=True,
# #                                     xi = .1)

if __name__ == '__main__':
    plt.close('all')
    curr_pos = 0
    plot_counter = 1
    plots = []
    plot_titles = []
    
    def key_event(e):
        global curr_pos
    
        if e.key == "right":
            curr_pos = curr_pos + 1
        elif e.key == "left":
            curr_pos = curr_pos - 1
        else:
            return
        curr_pos = curr_pos % len(plots)
    
        fig.clf()
        p = plots[curr_pos]
        ax = fig.add_subplot(111)
        ax.imshow(p)
        ax.set_title("Predicted Probability: "+(plot_titles[curr_pos]))
        fig.canvas.draw()
    
    main(0.5,30,5,1000,cityList,nCities)
    fig = plt.figure()
    plt.gcf().canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)
    ax.imshow(plots[0])
    ax.set_title("Predicted Probability: "+(plot_titles[0]))
    plt.show()
    



