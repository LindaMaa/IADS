import math
import graph
import random
import matplotlib.pyplot as plt
import numpy as np

# generate a non-metric graph
# input: n - number of nodes
def nonMetricGraph(n):

    for node1 in range(n):
        for node2 in range((node1 + 1), n):

            # create a new file
            if node1 == 0 and node2 == 1:
                edge = int(random.random() * n) + 1
                file = open(f'{n}nodes', "w")
                file.write(f' {node1} {node2} {edge}' + '\n')
                file.close()

            # append to an existing file
            else:
                edge = int(random.random() * n) + 1
                file = open(f'{n}nodes', "a")
                file.write(f' {node1} {node2} {edge}' + '\n')
                file.close()

# generate an euclidean graph
# inputs: n - number of nodes, xLimit - limiting value for x, yLimit - limiting value for y
def euclideanGraph(n, xLimit, yLimit):

    for node1 in range(n):
        node = (int(random.random() * xLimit), int(random.random() * yLimit))

        # create a new file
        if node1 == 0:
            file = open(f'cities{n}tests', "w")
            file.write(f' {node[0]} {node[1]}' + '\n')
            file.close()

        # append to an existing file
        else:
            file = open(f'cities{n}tests', "a")
            file.write(f' {node[0]} {node[1]}' + '\n')
            file.close()

# a function that produces a graph displaying cost on the y-axis and algorithm category on the x-axis
# inputs: n - number of nodes on the graph, euclidean - graph either euclidean or non-euclidean
def compare(numNodes, euclid):

    # create an euclidean graph with given parameters
    if euclid == True:
        euclideanGraph(numNodes, 2000, 2000)
        g = graph.Graph(-1, f'cities{numNodes}tests')

    # create a non-euclidean graph with given parameters
    else:
        nonMetricGraph(numNodes)
        g = graph.Graph(numNodes, f'{numNodes}nodes')

    # plot the graphs
    data = {}

    g.swapHeuristic()
    data["SwapHeuristic"] = g.tourValue()
    g.perm = [i for i in range(numNodes)]

    g.TwoOptHeuristic()
    data["TwoOptHeuristic"] = g.tourValue()
    g.perm = [i for i in range(numNodes)]

    g.Greedy()
    data["Greedy"] = g.tourValue()
    g.perm = [i for i in range(numNodes)]

    g.threeOptHeuristic()
    data["ThreeOptHeuristic"]=g.tourValue()
    g.perm = [i for i in range(numNodes)]

    g.myDynamicAlgorithm()
    data["MyDynamicAlgorithm"]=g.tourValue()
    g.perm = [i for i in range(numNodes)]

    # plot the results
    plt.plot(list(data.values()))
    plt.gca().set_xticks(np.arange(len(list(data.keys()))))
    plt.gca().set_xticklabels(list(data.keys()), rotation=15)
    plt.title(f'{numNodes} cities')
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig(f'{numNodes} Cities.png')


# calculate how close an algorithm is performing to the optimal solution
# save the performance of each algorithm into an array
# the closer to 1 the result is, the more optimal the algorithm is
def ratioToOptimal(numNodes, euclid):

     # create an euclidean graph with given parameters
     if euclid == True:
         euclideanGraph(numNodes, 100, 100)
         g = graph.Graph(-1, f'cities{numNodes}tests')

     # create a non-euclidean graph with given parameters
     else:
         nonMetricGraph(numNodes)
         g = graph.Graph(numNodes, f'{numNodes}nodes')

     ratios = {}

     g.swapHeuristic()
     ratios["SwapHeuristic"] = g.tourValue()/g.referenceAlgorithmStart()
     g.perm = [i for i in range(numNodes)]

     g.TwoOptHeuristic()
     ratios["TwoOptHeuristic"] = g.tourValue()/g.referenceAlgorithmStart()
     g.perm = [i for i in range(numNodes)]

     g.Greedy()
     ratios["Greedy"] = g.tourValue()/g.referenceAlgorithmStart()
     g.perm = [i for i in range(numNodes)]

     g.threeOptHeuristic()
     ratios["ThreeOptHeuristic"] = g.tourValue()/g.referenceAlgorithmStart()
     g.perm = [i for i in range(numNodes)]

     g.myDynamicAlgorithm()
     ratios["MyDynamicAlgorithm"] = g.tourValue()/g.referenceAlgorithmStart()
     g.perm = [i for i in range(numNodes)]

     print(ratios)


def main():
    # ratioToOptimal(10,False)
    compare(10, True)


if __name__ == '__main__':
    main()

