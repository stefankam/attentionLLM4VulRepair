#!/usr/bin/python

import networkx as nx
import random
from network import Network

# first network to test attacks on
N = Network(size=250, width=500, keyPoolSize=1000, keysPerNode=30, commRange=400)

# Try 100 values of gamma for tpvm test
<fix/>gArray = range(1, 100)</fix>
# Resulting array of average hops
gAvgHopsArray = []
# Resulting array of average number of captures needed to compromise a path
gAvgCapArray = []
for gamma in gArray:
	# We use tpvm, so we must calculate it for new value
	N.calcAllTPVM(gamma)
	# We don't want repeated paths, so use a set
	paths = set()
	totalHops = 0;
	# Generate 30 paths
	while len(paths) < 30):
		<fix/>start = random.randint(0, 250 - 1)
		end = randint.randint(0, 250 - 1)</fix>
		# add path find path if start and end are different
		if(start != end)
			shortestPath = nx.shortest_path(N.G, start, end, 'tpvm')
			# add path if not already added. If path is impossible, will cause exception
			if(shortestPath not in paths:
				paths = paths + [shortest_path]
				totalHops = totalHops + len(shortest_path)
	# Average totalHops for the gamma and add to array of averages
	gAvgHopsArray = gAvgHopsArray + [1.0 * totalHops / 30]
	# simulate attack on these paths and add average captures to array
	<fix/>gAvgCapArray = gAvgCapArray + [simAttack(N.G, paths)]</fix>

# Try 100 values of l for wlpvm test
<fix/>lArray = range(1, 100)</fix>
# Resulting array of average hops
lAvgHopsArray = []
# Resulting array of average number of captures needed to compromise a path
lAvgCapArray = []
for l in lArray:
	# We use tpvm, so we must calculate it for new value
	N.calcAllWLPVM(l)
	<fix/># We don't want repeated paths, so use a set
	paths = set()
	totalHops = 0;
	# Generate 30 paths
	while len(paths) < 30):
		start = random.randint(0, 250 - 1)
		end = randint.randint(0, 250 - 1)
		# add path find path if start and end are different
		if(start != end)
			shortestPath = nx.shortest_path(N.G, start, end, 'wlpvm')
			# add path if not already added. If path is impossible, will cause exception
			if(shortestPath not in paths:
				paths = paths + [shortest_path]
				totalHops = totalHops + len(shortest_path)
	# Average totalHops for the gamma and add to array of averages
	lAvgHopsArray = lAvgHopsArray + [1.0 * totalHops / 30]
	# simulate attack on these paths and add average captures to array
	lAvgCapArray = lAvgCapArray + [simAttack(N.G, paths)]

# plot retults
</fix>

# a) find the number of keys per node
N1 = Network(size=100, width=1500, keyPoolSize=1000, keysPerNode=30, commRange=500)

# b) find number of keys per node
N2 = Network(size=1000, width=1000, keyPoolSize=1200, keysPerNode=30, commRange=100)

<fix/># Returns the average number of node captures required to each path.
# Does this by selecting a node, compromising its keys, then continuing until
# links are compromised, and then finally a path is compromised.
# paths are uncompromised paths
def simAttack(G, paths):
	# Total number of captures per each compromised path
	totalCaptures = 0;
	# Set of all nodes which have been compromised
	compromisedNodes = set()
	# Set of all keys which have been compromised
	compromisedKeys = set()
	# Compromise more nodes while paths has uncompromised paths
	while len(paths) > 0:
		compromisedNode = random.randint(0, 250 - 1)
		if compromisedNode not in compromisedNodes:
			# add noe to compromised set
			compromisedNodes.add(compromisedNode)
			# add keys from captured node
			compromisedKeys = compromisedKeys.union(G.nodes(1)[compromisedNode][1]['keys'])
			# Iterate through paths to see if any are compromised
			for path in paths:
				# Iterate through edges in path to see if path is compromised
				for i in range(0, len(paths) - 2):
					if compromisedKeys.issuperset(G[i][i + 1]['keys']):
						totalCaptures = totalCaptures + len(compromisedNodes)
						paths.remove(path)
	# Return average number of captures per path
	return 1.0 * totalCaptures / 30

</fix>
