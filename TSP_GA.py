import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import time
import math
from skimage.transform import resize
#from PIL import image as Image


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
    #print(tour)
    #print(type(tour))
    z = D[(tour[-1], tour[0])]    # edge from last to first city of the tour
    for i in range(1,len(tour)):
        z += D[tour[i], tour[i-1]]      # add length of edge from city i-1 to i
    return z

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
            D[i,j] = distL2(x1,y1,x2,y2)
            D[j,i] = D[i,j]
    ## Epsilon to set small floating points to zero
    np.fill_diagonal(D, 0)
    return D

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
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
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
   # print(len(population))
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
        #print('\n',i,'\n',population[i])
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, eliteSize, currentGen):
    #print("Selection call")
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
    #print("call to breed function")
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2

    # f, (a1,a2,a3) = plt.subplots(ncols=3, squeeze=True)
    # p1= [a.to_tuple() for a in parent1]
    # p2= [a.to_tuple() for a in parent2]
    # c = [c.to_tuple() for c in child]
    # #print(br)
    # a1.scatter([b[0] for b in p1],[c[1] for c in p1])
    # a1.plot([a[0] for a in p1],[a[1] for a in p1])
    # a1.set_title("Parent 1")
    # a2.scatter([a[0] for a in p2],[a[1] for a in p2])
    # a2.plot([a[0] for a in p2],[a[1] for a in p2])
    # a2.set_title("Parent 2")
    # a3.scatter([a[0] for a in c],[a[1] for a in c])
    # a3.plot([a[0] for a in c],[a[1] for a in c])
    # a3.set_title("Child")
    # f.suptitle("Example of Crossover Operation")
    # f.show()
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
    
    #print('\n\n',pi_mod,indices)
    #Investigate individual model predictions
    # print("\nTour A: ", A.array_form)
    # print("Tour B: ", B.array_form)
    # print("Tour A length: \t", length(A.array_form, d_matrix))
    # print("Tour B length: \t", length(B.array_form, d_matrix))
    # print("Predicted value from model: \t", predicted)
    if length(pi_mod,d_matrix) < length(indices,d_matrix):
        return [individual[i] for i in pi_mod]
    else:
        return individual

def mutatePopulation(population, mutationRate,d_matrix):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate,d_matrix)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate,d_matrix):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize, currentGen)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate,d_matrix)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations, d_matrix):
    time_start = time.time()
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    print("Initial distance: \t" + str(1 / rankRoutes(pop)[0][1]))
    best=99999999999
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate, d_matrix=d_matrix)
        progress.append(1 / rankRoutes(pop)[0][1])
    print("Final distance: \t" + str(1 / rankRoutes(pop)[0][1]))
    n_best = 1 / rankRoutes(pop)[0][1]
    if n_best < best:
        best=n_best
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    #f,(a1,a2) = plt.subplots(1,2)
    line, = plt.plot(progress, label = "Off the shelf GA")
    # a1.plot(progress, label = "Off the shelf GA")
    # #legend2 = plt.legend(handles=[line], loc='lower right')
    # #plt.gcs.add_artist(legend2)
    # # a1.('Distance of best route found')
    # # a1.xlabel('Generations')
    # #print(type(bestRoute))
    # br= [a.to_tuple() for a in bestRoute]
    # br.insert(0,bestRoute[-1].to_tuple())
    # print(br)
    # #print(br)
    # a2.scatter([a[0] for a in br],[a[1] for a in br])
    # a2.plot([a[0] for a in br],[a[1] for a in br])
    # a2.set_title("Best tour found")
    # #print("Shortest route found: ", best)
    # #plt.plot(progress)
    # f.show()
    time_finish=time.time()
    time_e =  time_finish-time_start
    print("Time elapsed: \t",(time_e))
    return line, bestRoute, time_e, best


def main(mutationRate, popSize, eliteSize, generations, cityList,nCities):
    # cityList = []
    
    # for i in range(0,nCities):
    #     cityList.append(City(index=i,x=int(random.random() * 100), y=int(random.random() * 100)))
    
    # # Get cities and normalized distance matrix for input into model
    city_pts = [i.to_tuple() for i in cityList]
    d_matrix = mk_matrix(city_pts)
    # #normalized_dm = normalize(d_matrix)
    
    r=geneticAlgorithm(population=cityList, popSize=popSize, eliteSize=eliteSize, mutationRate=mutationRate, generations=generations, d_matrix=d_matrix)
    return(r)
# time_finish=time.time()
# print("Time elapsed: \t",(time_finish-time_start))
# nCities=60
# cityList = []
    
# for i in range(0,nCities):
#     cityList.append(City(index=i,x=int(random.random() * 100), y=int(random.random() * 100)))
# time0 = time.time()
# main(cityList,nCities) 
# time1 = time.time() 
# print("time elapsed: ", time1-time0)
