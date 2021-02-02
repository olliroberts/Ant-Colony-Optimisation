# %% Import Libraries
import numpy as np

# %% Ant Colony Optimisation Class

class ant_colony:
   
    def __init__(self, nodes, q, evaporation, alpha, beta, n_ants, iterations):
        '''
        Create an instance of ant colony optimisation.

        Parameters
        ----------
        nodes : dict
            Dictionary of nodes and their coordinate positions.
        q : float
            Constant used during local phermeone calculation.
        evaporation : float
            Evaporation constant rho used when updating pheremone matrix.
        alpha : float
            Constant to control the impact of the pheremone levels in the
            probability computation.
        beta : float
            Constant to control the effect of local preference matrix H.
        n_ants : int
            Number of ants used per iteration.
        iterations : int
            Number of iterations.
            
        '''

        self.nodes = nodes
        self.q = q
        self.evaporation = evaporation
        self.alpha = alpha
        self.beta = beta
        self.n_ants = n_ants
        self.iterations = iterations
        
    def get_distance(self, start, end):
        '''
        Function that calculates the distance between two nodes.

        Parameters
        ----------
        start : tuple
            Coordinates of start node.
        end : tuple
            Coordinates of end node.

        Returns
        -------
        dist : float
            Straight line distance between start and end node
            calculated by pythagorous' theorm.
            
        '''
        
        x_dist = abs(start[0] - end[0])
        y_dist = abs(start[1] - end[1])
        dist = np.sqrt(x_dist) + np.sqrt(y_dist)
        return dist

    def get_adjacency(self):
        '''
        Function to intialise the adjacency and eta matrices,
        calculated from the input dictionary of nodes.

        Returns
        -------
        None. Assigns the two matrices to class methods.
        
        '''
        
        nodes = self.nodes
        adjacency = np.zeros((len(nodes), len(nodes)))
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                adjacency[i,j] = self.get_distance(list(nodes.values())[i],
                                                   list(nodes.values())[j])
        self.adjacency = adjacency
        self.eta = np.divide(1, adjacency, out=np.zeros_like(adjacency),
                             where=adjacency!=0)
        
    def get_total_distance(self, route):
        '''
        Function to calculate the total distance of a given route.

        Parameters
        ----------
        route : list of strings
            Ordered list of nodes to be visited in the current route.

        Returns
        -------
        total_distance : float
            The total distance of the current route in the same units
            as the input node coordinates.

        '''
        
        nodes = self.nodes
        dist = 0
        for i in range(len(route)):
            current_node_idx = list(nodes.keys()).index(route[i])
            try:
                next_node_idx = list(nodes.keys()).index(route[i+1])
            except:
                next_node_idx = list(nodes.keys()).index(route[0])
            dist += self.adjacency[current_node_idx][next_node_idx]
        total_distance = dist
        return total_distance
                
    def initialise_pheremone(self):
        '''
        Function to initialise the pheremone matrix

        Returns
        -------
        None. Assigns the matrix to a class method.

        '''
        
        pheremone = np.ones((self.adjacency.shape[0],
                             self.adjacency.shape[0]))
        np.fill_diagonal(pheremone, 0)
        self.pheremone = pheremone
        
    def initialise_route(self):
        '''
        Function to intialise the route. For each ant, this includes
        selecting the starting node and generating a list of unvisited nodes.

        Returns
        -------
        None. Assigns the start node, and unvisited nodes to class methods. 

        '''
        
        self.unvisited_nodes = list(self.nodes)
        self.start_node = np.random.choice(self.unvisited_nodes)
        self.start_node_idx = self.initial_route.index(self.start_node)
        self.unvisited_nodes.remove(self.start_node)
                
    def get_probabilities(self, start_node_idx, unvisited_nodes):
        '''
        Function to calculate the probabilities used in selecting
        the next node to visit. 

        Parameters
        ----------
        start_node_idx : int
            Index of the current node in the input node dictionary. This means
            the correct value can be found in the adjacency and eta matrices. 
        unvisited_nodes : list of strings
            List of nodes that have not been visited by the current ant.

        Returns
        -------
        None. Assigns an array of probabilities that matches the
        length of unvisited nodes as a class method.

        '''
        
        probabilities = []
        for node in range(len(unvisited_nodes)):
            next_node = unvisited_nodes[node] 
            next_node_idx = self.initial_route.index(next_node)
            denominator = 0
            for i in range(len(unvisited_nodes)):
                n = unvisited_nodes[i]
                n_idx = self.initial_route.index(n)
                denominator += (self.pheremone[start_node_idx][n_idx]
                                **self.alpha
                                *self.eta[start_node_idx][n_idx]
                                **self.beta)
            probability = ((self.pheremone[start_node_idx][next_node_idx]
                            **self.alpha
                            *self.eta[start_node_idx][next_node_idx]
                            **self.beta)/denominator)
            probabilities.append(probability)
        self.probabilities = probabilities #np.array([probabilities])
                            
    def optimise(self):
        '''
        Main class method. This function can be called from an ant colony
        optimisation object and returns the best route and distance. 

        Returns
        -------
        route_best : numpy array of strings
            The shortest route found as determined by the
            function get_total_distance().
        distance_best : float
            The total distance of route_best

        '''
        
        self.get_adjacency()
        self.initialise_pheremone()
        self.initial_route = list(self.nodes)
        self.route_best = self.initial_route
        self.distance_best = self.get_total_distance(self.route_best) 
        for i in range(iterations):
            local_pheremone = np.zeros((self.adjacency.shape[0],
                                        self.adjacency.shape[0]))
            for b in range(n_ants):
                self.initialise_route()
                route = list(self.start_node)
                current_node_idx = self.start_node_idx
                while len(self.unvisited_nodes) != 0:
                    self.get_probabilities(current_node_idx,
                                           self.unvisited_nodes)
                    next_node = np.random.choice(self.unvisited_nodes,
                                                 p=self.probabilities)
                    route.append(next_node)
                    current_node_idx = self.initial_route.index(next_node)
                    self.unvisited_nodes.remove(next_node)
                distance = self.get_total_distance(route)
                for node in range(len(route)):
                    current_node = route[node]
                    i = self.initial_route.index(current_node)
                    try:
                        next_node = route[node+1]
                    except:
                        next_node = route[0]
                    j = self.initial_route.index(next_node) 
                    new_pheremone = np.zeros((self.adjacency.shape[0],
                                              self.adjacency.shape[0]))
                    new_pheremone[i][j] = self.q/distance
                if distance < self.distance_best:
                    self.route_best, self.distance_best = route, distance
                local_pheremone = local_pheremone + new_pheremone
            self.pheremone = ((1-self.evaporation)*self.pheremone
                              + local_pheremone)
        return self.route_best, self.distance_best
        

# %% Create Ant Colony Problem and Solve
'''
Set nodes and parameters. Then create an instance of the ant colony.
Then use the optimise() class method to solve.

'''

nodes = {'A':(0,0), 'B':(10,4), 'C':(5,3), 'D':(7,12), 'E':(4,5)}
q = 50
evaporation = 0.5
alpha = 0.5
beta = 1.2
n_ants = 10
iterations = 10

problem = ant_colony(nodes, q, evaporation, alpha, beta, n_ants, iterations)
solution = problem.optimise()
print('The best route is', solution[0])
print('with a total distance of:', np.round(solution[1], 4), 'units')
