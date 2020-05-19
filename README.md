# TSP
Library to hold implementation work for Traveling Salesman Problem research for my senior thesis at Whitman College, May 2020.


# Visual Agents
Convolutional Neural Network models trained to identify optimal TSP tours
<img src="TSP_NVA.gif" width="404">
<img src="TSP_NVA1.gif" width="404">

# NEAT Reinforcement Learning Simulation
Applying the NEAT (Neuroevolution of Augmenting Topologies) algorithm of neural network training to Recurrent Neural Networks controlling the simulated movement of TSP salesman. At each stage of the simulation, each RNN receives the current state as input and predicts a city for its associated salesman to visit. Following a reinforcement learning paradigm, the salespeople are rewarded based on the number of cities visited within the alloted distance. In the sample below, you can see the improvement from 11 to 12 to 13 cities in the first few generations of the NEAT algorithm.
<img src="TSPReinforcementNEAT.gif" width="800">
