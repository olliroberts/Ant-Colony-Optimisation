# Ant-Colony-Optimisation
Python implementation of the Ant Colony Optimisation algorithm. 
https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms

### Requirements
This implementation used the NumPy library. 

### How to use:
Create a dictionary of nodes and their coordinates.
```
nodes = {'A':(0,0), 'B':(10,4), 'C':(5,3), 'D':(7,12), 'E':(4,5)}
```
Then set all the algorithm hyperparameters.
```
q = 50
evaporation = 0.5
alpha = 0.5
beta = 1.2
n_ants = 10
iterations = 10
```
Finally create an instance of the ant colony and call the optimise() class method to solve.
```
problem = ant_colony(nodes, q, evaporation, alpha, beta, n_ants, iterations)
solution = problem.optimise()
print('The best route is', solution[0])
print('with a total distance of:', np.round(solution[1], 4), 'units')
```
The output is as follows:
```
The best route is ['A', 'E', 'D', 'B', 'C']
with a total distance of: 20.3785 units
```
