import math
import sys

# helper function: calculate euclidean distance between points p and q
def euclid(p, q):
    x = p[0] - q[0]
    y = p[1] - q[1]
    return math.sqrt(x * x + y * y)

class Graph:

    # parse two kinds of graphs based on the value of n
    # n= -1 case: read points in the Euclidean plane -> Euclidean Graph
    # n>0 case: a general graph in a different format -> non-Euclidean Graph
    # self.perm, self.dists, self.n are the key variables to be set up
    def __init__(self, n, filename):

        # general case -> assuming general TSP input
        file = open(filename, "r")
        self.n = n

        # euclidean TSP -> count the number of lines in the file and assign it to self.n
        if (n == -1):
            num_lines = 0
            for line in file:
                num_lines = num_lines + 1
            self.n = num_lines

        # initialise self.dist table
        self.dist = [[0 for col in range(self.n)] for row in range(self.n)]

        if n == -1:
            # parse lines from f
            coordinates_table = []
            file = open(filename, "r")
            for line in file:
                coordinates_table.append(line.split())
                # cast distances to integers
                for j in range(2):
                    coordinates_table[-1][j] = int(coordinates_table[-1][j])

            # fill in the self.dist table with distances
            for i in range(len(coordinates_table)):
                for j in range(len(coordinates_table)):
                    self.dist[i][j] = euclid(coordinates_table[i], coordinates_table[j])
        else:
            # parse lines from f
            coordinates_distance_table = []
            file = open(filename, "r")
            for line in file:
                coordinates_distance_table.append(line.split())

            # fill in the self.dist table with weights
            for line in range(len(coordinates_distance_table)):
                self.dist[int(coordinates_distance_table[line][0])][int(coordinates_distance_table[line][1])] = int(
                    coordinates_distance_table[line][2])
                self.dist[int(coordinates_distance_table[line][1])][int(coordinates_distance_table[line][0])] = int(
                    coordinates_distance_table[line][2])

        # initialise self.perm with the identity permutation
        self.perm = [i for i in range(self.n)]

        # 2D array for reference DP algorithm for part D
        # set a limit
        if self.n <= 15:
            self.pathLength = [[(-1) for j in range(1 << self.n)] for i in range(self.n)]


    # calculate the cost of the tour (represented by self.perm)
    def tourValue(self):

        total_cost = 0

        for i in range(len(self.perm)):
            j = i + 1
            # calculate the cost between 2 adjacent points in self.perm (i and its neighbor j)
            total_cost += self.dist[self.perm[i]][self.perm[j % len(self.perm)]]
        return total_cost


    # attempt the swap of cities i and i+1 in self.perm
    # commit to the swap if it improves the cost of the tour.
    # return True/False depending on success
    def trySwap(self, i):

        cost_before_swap = self.tourValue()

        # swap cities i and i+1 in self.perm
        temp = self.perm[i]
        self.perm[i] = self.perm[(i + 1) % self.n]
        self.perm[(i + 1) % self.n] = temp

        cost_after_swap = self.tourValue()

        # if swap does not improve cost, swap cities again (undo the effect) else keep the swap
        if cost_after_swap > cost_before_swap:
            temp = self.perm[i]
            self.perm[i] = self.perm[(i + 1) % self.n]
            self.perm[(i + 1) % self.n] = temp
            return False

        else:
            return True


    # consider the effect of reversing the segment between
    # self.perm[i] and self.perm[j], and commit to the reversal
    # if it improves the tour value
    # return True/False depending on success.
    def tryReverse(self, i, j):

        # calculate start and end of the sequence that will be reversed
        start = self.perm[(i - 1) % self.n]
        end = self.perm[(j + 1) % self.n]

        # compute cost before and after reversal
        dist_before = self.dist[start][self.perm[i]] + self.dist[self.perm[j]][end]
        dist_after = self.dist[start][self.perm[j]] + self.dist[self.perm[i]][end]

        # if cost is improved then reverse else keep the same
        if dist_after < dist_before:
            self.perm[i:j + 1] = self.perm[i:j + 1][::-1]
            return True

        else:
            return False


    # given function to perform trySwap on the self.perm
    def swapHeuristic(self):
        better = True
        while better:
            better = False
            for i in range(self.n):
                if self.trySwap(i):
                    better = True


    # given function to perform tryReverse on the self.perm
    def TwoOptHeuristic(self):
        better = True
        while better:
            better = False
            for j in range(self.n - 1):
                for i in range(j):
                    if self.tryReverse(i, j):
                        better = True


    #  the Greedy heuristic which builds a tour starting
    # from node 0, taking the closest (unused) node as 'next' each time
    def Greedy(self):

        # start from node 0
        self.perm[0] = 0

        # keep track of the unused nodes
        unused_nodes = []
        for node in range(1, self.n):
            unused_nodes.append(node)

        # find the nearest neighbor(j) for each node(i) and assign it to the best_node
        for i in range(self.n - 1):
            best_node = 0
            best_distance = 1000000000000000000
            for j in unused_nodes:
                if (self.dist[self.perm[i]][j] < best_distance):
                    best_node = j
                    best_distance = self.dist[self.perm[i]][j]

            # after nearest node is added, remove it from unused nodes and continue
            unused_nodes.remove(best_node)
            self.perm[i + 1] = best_node


    # try to reverse self.perm[x:y], if the cost improves, reverse
    def tryReverse3Opt(self, x, y, z):

        #take points to form segments
        p1 = self.perm[x - 1]
        p2 = self.perm[x]
        p3 = self.perm[y - 1]
        p4 = self.perm[y]
        p5 = self.perm[z - 1]
        p6 = self.perm[z % len(self.perm)]

        # calculate distances for different arrangements
        try0 = self.dist[p1][p2] + self.dist[p3][p4] + self.dist[p5][p6]
        try1 = self.dist[p1][p3] + self.dist[p2][p4] + self.dist[p5][p6]
        try2 = self.dist[p1][p2] + self.dist[p3][p5] + self.dist[p4][p6]
        try3 = self.dist[p1][p4] + self.dist[p5][p2] + self.dist[p3][p6]
        try4 = self.dist[p6][p2] + self.dist[p3][p4] + self.dist[p5][p1]

        # find the lowest-cost tour for given sub-tours
        if try0 > try1:
            self.perm[x:y] = reversed(self.perm[x:y])
            return -try0 + try1

        elif try0 > try2:
            self.perm[y:z] = reversed(self.perm[y:z])
            return -try0 + try2

        elif try0 > try4:
            self.perm[x:z] = reversed(self.perm[x:z])
            return -try0 + try4

        elif try0 > try3:
            temp = self.perm[y:z] + self.perm[x:y]
            self.perm[x:z] = temp
            return -try0 + try3

        return 0


    # create all possible edge combinations for segments
    # a helper function used in 3Opt algorithm
    def generateSegments(self, n):
        return ((x, y, z)
                for x in range(self.n)
                for y in range(x + 2, self.n)
                for z in range(y + 2, self.n + (x > 0)))


    # run 3Opt algorithm on self.perm for all possible segment combinations
    def threeOptHeuristic(self):
        while True:
            difference = 0

            for (seg1, seg2, seg3) in self.generateSegments(self.n):
                difference = difference + self.tryReverse3Opt(seg1, seg2, seg3)

            if difference >= 0:
                break


    # a helper function that calculates the cost of a tour given by a sequence (for variable lengths from 1 to n)
    # used in the implementation of myDynamicAlgorithm in part C
    def tourValueMyAlgorithm(self, sequence):

        total_cost = 0

        for i in range(len(sequence)):
            j = i + 1
            # calculate the cost between 2 adjacent points in sequence
            total_cost += self.dist[sequence[i]][sequence[j % len(sequence)]]
        return total_cost


    # starting at node start, the algorithm builds up the solution optimising for the lowest cost at every addition of a new node
    # the length of the sequence increases by 1 in every iteration until len(sequence)=self.n
    def dynamicAlgorithm(self, start):

        sequence = []  # store the lowest-cost permutation
        sequence.append(start)

        # keep adding nodes until full length
        for node in range(0, self.n):
            best_sequence = []
            best_sequence_cost = 100000000000000000000000000
            if not (node in sequence):

                # try to insert node n at all positions available
                for place in range(node + 1):
                    sequence.insert(place, node)

                    # insert node n at the place that will result in the lowest cost
                    if self.tourValueMyAlgorithm(sequence) < best_sequence_cost:
                        best_sequence = sequence[:]
                        best_sequence_cost = self.tourValueMyAlgorithm(sequence)

                    sequence.pop(place)

                sequence = best_sequence

                #best sequence found for a given starting node is assigned to self.perm
                self.perm = best_sequence


    # try dynamicAlgorithm with all nodes as starting nodes and optimise for the lowest cost
    def myDynamicAlgorithm(self):

        best_cost = 10000000000000000
        best_perm = [] # store the lowest cost permutation

        # try all nodes as starting nodes - n
        for node in range(self.n):
            self.dynamicAlgorithm(node)
            tryMe = self.perm

            # select the starting node that will result in the lowest cost permutation
            if self.tourValueMyAlgorithm(tryMe) < best_cost:
                best_cost = self.tourValueMyAlgorithm(tryMe)
                best_perm = tryMe

        self.perm = best_perm


   # NOTE: This algorithm is only used for finding the optimal solution for small inputs in part D
   # inspiration from: https://codingblocks.com/resources/travelling-salesman/
   # recursive DP algorithm (exponential time)
   # starting node 0
   # Parameters:
   # visited - range 0-(2^n-1), binary number denoting if a node is visited
   # pos - the node/city which is visited last
   # start - the start node

    def referenceAlgorithm(self, visited=1, position=0, start=0):

        # only allow small inputs
        if self.n > 15:
            return False

        allVisitedNodes = (1 << self.n) - 1

        # base case
        if (visited == allVisitedNodes):
            return self.dist[position][start]

        if (self.pathLength[position][visited] != -1):
            # already calculated cost
            return self.pathLength[position][visited]

        cost = sys.maxsize

        for node in range(self.n):
            # check if a certain node is visited
            if (visited & (1 << node) == 0):
                new_cost = self.dist[node][position] + self.referenceAlgorithm((visited | (1 << node)), node, start)
                cost = min(cost, new_cost)
        self.pathLength[position][visited] = cost
        return cost

    # find optimal solution
    # by finding the best starting node
    # returns the minimal cost
    def referenceAlgorithmStart(self):

        # only works for small inputs
        if self.n > 15:
            return False

        minimum_cost = sys.maxsize
        for i in range(self.n):
            currrent_cost = self.referenceAlgorithm((1 << i), i, i)
            if minimum_cost > currrent_cost:
                minimum_cost = currrent_cost
        return minimum_cost

if __name__ == "__main__":
    g = Graph(-1, "cities10t")
    print(g.referenceAlgorithmStart())
    g.myDynamicAlgorithm()
    print(g.tourValue())
    #print(g.tourValue())
    #g.TwoOptHeuristic()
    #print(g.tourValue())
    #g.Greedy()
    #print(g.tourValue())
