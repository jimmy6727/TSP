import pygame
import numpy as np
import math
import concorde.tsp as concorde
import os
import neat
from sklearn.preprocessing import normalize
from keras.utils import to_categorical
import time
# Set initial generation value
gen = 0

BACKGROUND_IMG = pygame.transform.scale(pygame.image.load("background.png"),(500,500))
GRAD_IMG = pygame.transform.scale(pygame.image.load("grad.png"),(170,20))
pygame.font.init()  # init font
    
STAT_FONT = pygame.font.SysFont("comicsans", 20)
GR_FONT = pygame.font.SysFont("comicsans", 15)
BEST_FONT = pygame.font.SysFont("comicsans", 15)
END_FONT = pygame.font.SysFont("comicsans", 20)
DRAW_LINES = False

pygame.font.init()  # init font
WIN_WIDTH = 500
WIN_HEIGHT = 500
pygame.init()
window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("AI Traveling Salesperson")
if os.path.exists('outputs.txt'):
    os.remove('outputs.txt')
file = open('outputs.txt', 'w+')


class Salesperson(pygame.sprite.Sprite):
    
    ANIMATION_TIME = 5
    
    def __init__(self, index, x, y, waypoints, distance_available):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.x=x
        self.y=y
        self.xvel = 0
        self.index = index
        self.visited_city = None
        self.yvel = 0
        self.tick_count = 0
        self.best = False
        self.img_count = 0
        self.img = pygame.Surface((5,5))
        self.img.fill(pygame.Color(90,130,145))
        self.rect = self.img.get_rect()
        self.speed = 5
        self.waypoints=waypoints
        self.unvisited=waypoints.copy()
        self.visited = []
        self.distance_available = distance_available
        

    def update_color(self,value):
        # Map fitness value from 1-1000 to color value from 0-255
        # value = value*100
        g = int(value//15)
        r = int(value//5-40)
        b = 100
        if g < 0:
            g=0
        if g > 255:
            g=255
        if r < 0:
            r=0
        if r > 255:
            r=255
        if b < 0:
            b=0
        if b > 255:
            b=255
            
        self.img.fill(pygame.Color(r,g,b))
    
    def move(self):
        #Move randomly to test
        # d_x = np.random.uniform(-1, 1, 1)
        # d_y = np.random.uniform(-1, 1, 1)
        
        
        # Convert to unit vector
        mag = math.sqrt(self.x_dir**2+self.y_dir**2)
        if mag == 0:
            d_x = 0
            d_y = 0
        else:
            d_x = self.x_dir/mag
            d_y = self.y_dir/mag
        
        self.x += d_x*self.speed
        self.y += d_y*self.speed
        
    def draw(self, window):
        """
        draw the salesperson
        :param win: pygame window or surface
        :return: None
        """
        self.img_count += 1
        window.blit(self.img, (self.x, self.y))
        if self.best:
            t_label = STAT_FONT.render("Best -> ",True,(0,0,0))
            window.blit(t_label, (self.x-45, self.y-5))
   
    def update_distance(self):
        self.distance_available -= self.speed
        
    def update_direction(self, x_dir, y_dir):
        self.x_dir = x_dir
        self.y_dir = y_dir

    
class City(pygame.sprite.Sprite):

    def __init__(self, index, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.index = index
        self.x=x
        self.collision_buffer = 0
        self.y=y
        self.img = pygame.Surface((5,5))
        self.img.fill(pygame.Color(0,0,0))
        #self.red = RED_CITY
        self.rect = self.img.get_rect()
        self.visited = False

    def draw(self, window):

        window.blit(self.img, (self.x, self.y))

            
    
def draw_window(window, salespeople, cities, optimal_tour, fitness, distance_used):

    # background
    window.blit(BACKGROUND_IMG, (0,0))
    window.blit(GRAD_IMG, (300,35))
    a1 = GR_FONT.render("Lower",True,(0,0,0))
    window.blit(a1, (300, 24))
    a2 = GR_FONT.render("Higher",True,(0,0,0))
    window.blit(a2, (440, 24))
    # cities
    for city in cities:
        city.draw(window)
        
    # salesperson
    for s in salespeople:
        s.draw(window)    
    
    # optimal tour value
    opt_label = STAT_FONT.render("Optimal Tour: " + str(optimal_tour),True,(0,0,0))
    window.blit(opt_label, (10, 5))
    
    # fitness
    fit_label = STAT_FONT.render("Best Fitness Score: " + str(fitness),True,(0,0,0))
    window.blit(fit_label, (300, 5))

    # # generations
    # gen_label = STAT_FONT.render("Gens: " + str(gen-1),1,(0,0,0))
    # window.blit(gen_label, (10, 50))
    
    # generations
    t_label = STAT_FONT.render("Distance remaining: " + str(distance_used),True,(0,0,0))
    window.blit(t_label, (10, 30))
    
    
    pygame.display.update()

def get_network_inputs(city_group, salesperson):
    input_list = []
    for c in city_group:
        if c.visited == True:
            input_list.append(0)

        else:
            distance = math.sqrt((salesperson.x-c.x)**2+(salesperson.y-c.y)**2)+100
            input_list.append(distance)

    return list(normalize(np.array(input_list).reshape(1,-1))[0])

def eval_genomes(genomes, config):
    
    # Initialize salesperson and cities
    city_group = pygame.sprite.Group()
    for c in mCities:
        city_group.add(c)
        
    waypoints = [(c.x,c.y) for c in city_group]
    #print("length waypoints :", len(waypoints))
    
    # Arbitrary starting city
    c0 = waypoints[0]
    x0=c0[0]
    y0=c0[1]
    
    salespeople = []
    nets = []
    ge = []
    
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.RecurrentNetwork.create(genome, config)
        nets.append(net)
        salespeople.append(Salesperson(genome_id,x0,y0,waypoints,optimal_tour))
        ge.append(genome)
    
    clock = pygame.time.Clock()

    while len(salespeople) > 10:
        
        clock.tick(100)
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                #sys.quit()
                break
            
        # Run one iteration of game loop
            
        for s in salespeople:
            # Check for city visit
            s.visited_city = None
            for c in city_group:
                # get distance from salesperson to city
                d = math.sqrt((c.x-s.x)**2+(c.y-s.y)**2)
                # if we have a collision
                if d <= 4:
                    # Penalty for repeat visits
                    if (c.x,c.y) in s.visited:
                        #print("\n Repeat collision between city ",c.index," at ", c.x,c.y, " and salesperson ", salespeople.index(s), " at ", s.x,s.y)
                        ge[salespeople.index(s)].fitness -=50
                        #print("salesperson ", salespeople.index(s), " visited list: ", str(s.visited))
                        s.visited_city = c
                    # Reward for new city visits
                    if (c.x,c.y) in s.unvisited:
                        #print("\n New collision between city ",c.index," at ", c.x,c.y, " and salesperson ", salespeople.index(s), " at ", s.x,s.y)
                        s.visited.append((c.x,c.y))
                        s.unvisited.pop(s.unvisited.index((c.x,c.y)))
                        #print("salesperson ", salespeople.index(s), " visited list: ", str(s.visited))
                        # Give the salesperson a one time reward
                        ge[salespeople.index(s)].fitness += 100
                        s.visited_city = c
    
            if s.visited_city != None:
                # Get current state:  total distance, x distance
                # and y distance to every city
                state = get_network_inputs(city_group, s)
        
                
                # Network picks an (x,y) direction to go (make sure config file has 2 outputs in this case)
                    # prediction = agent.model.predict(state)
                    # x=prediction[0]
                    # y=prediction[1]
                
                # Send state to RNN
                # Network picks a city via 10-way softmax. Get network output and go to city
                prediction = nets[salespeople.index(s)].activate(state)
                #print(prediction)
                #softmax_output = to_categorical(np.argmax(prediction), num_classes=20)
                #print(softmax_output)
                target_city = None
                counter = 0
                while target_city == None:
                    predicted_city = prediction.index(max(prediction))
                    if s.waypoints[predicted_city] in s.unvisited:
                        target_city = s.waypoints[predicted_city]
                    else:
                        prediction[predicted_city] = 0
                #print("predicted city index = ",predicted_city)
                #print("prediction of shape", prediction.shape," : ", prediction)
                
                x = target_city[0]-s.x
                y = target_city[1]-s.y
                    
                s.update_direction(x, y)
              
                if s.index == 1: 
                    #Output file for analysis
                    file.write("\n\n\nsalesman position: "+ str(s.x)+','+str(s.y))
                    file.write("\nnetwork prediction: " +str(prediction))
                    file.write("\npredicting city :" +str(predicted_city)+' at position '+str(target_city[0])+','+str(target_city[1]))
                    file.write("\nDistance to target city: "+str(x)+','+str(y))
                    file.write("\nHeading in direction "+str(x)+','+str(y))
                

            s.move()   
            s.update_distance()
            s.update_color(ge[salespeople.index(s)].fitness)
            s.best = False
        
        # best fitness
        best = 0
        for g in ge:
            if g.fitness > best:
                best=g.fitness
                indx = ge.index(g)
        salespeople[indx].best = True  
        draw_window(window, salespeople, city_group, optimal_tour, fitness = best, distance_used=s.distance_available)
       
        global h
        if optimal_tour - s.distance_available <=10 and h == False:
            h = True
            time.sleep(7)
            
        for s in salespeople:      
                
            # Die 
            if s.distance_available <= 0:
                nets.pop(salespeople.index(s))
                ge.pop(salespeople.index(s))
                salespeople.pop(salespeople.index(s))
            
        # if they go off the game screen
        if len(salespeople) > 0:
            for s in salespeople:
                if s.x > 500 or s.y > 500:
                    nets.pop(salespeople.index(s))
                    ge.pop(salespeople.index(s))
                    salespeople.pop(salespeople.index(s))
                
def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to control salespeople
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 150)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    
    # Generate and instantiate cities
    numCities = 30
    mCities = []
    h = False
    
    x_values = np.random.randint(50,450,size=numCities)
    y_values = np.random.randint(80,450,size=numCities)
    
    #Write city data to tsp file and solve with concorde
    name='test1'
    filename='MyData/coords/test1.tsp'
    fp=open(filename, 'w+')
    concorde.write_tsp_file(fp, x_values, y_values, "EUC_2D", name)
    fp.close()
    solver = concorde.TSPSolver.from_tspfile(filename)        
    solution=solver.solve(verbose=False)
    optimal_tour = solution.optimal_value
    
    for i in range(numCities):
        mCities.append(City(i,x_values[i],y_values[i]))

    
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.feedforward.txt')
    run(config_path)
    
    
    
    
 