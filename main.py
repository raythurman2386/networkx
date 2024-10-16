import networkx as nx
import matplotlib.pyplot as plt

# Example 1: Adding and Removing Nodes and Edges
H = nx.Graph()
H.add_nodes_from([1, 2, 3, 4, 5])
H.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])

print("Nodes of graph H:")
print(H.nodes())
print("Edges of graph H:")
print(H.edges())

# Remove a node and an edge
H.remove_node(5)
H.remove_edge(1, 2)

# Draw the modified graph
nx.draw(H, with_labels=True)
plt.show()

# Example 2: Working with Different Types of Graphs
# Creating a directed graph
D = nx.DiGraph()
D.add_edges_from([(1, 2), (2, 3), (3, 1)])

print("Nodes of directed graph D:")
print(D.nodes())
print("Edges of directed graph D:")
print(D.edges())

# Draw the directed graph with node labels
nx.draw(D, with_labels=True, node_color='lightblue', arrows=True)
plt.show()

# Example 3: Adding Attributes to Nodes and Edges
G = nx.Graph()
G.add_node(1, label='A')
G.add_node(2, label='B')
G.add_edge(1, 2, weight=4.2)

# Printing node and edge attributes
print("Node attributes in graph G:")
print(G.nodes(data=True))
print("Edge attributes in graph G:")
print(G.edges(data=True))

# Example 4: Analyzing Graph Properties
G = nx.cycle_graph(4)  # Creates a cycle graph with 4 nodes
print("Degree of each node in graph G:")
print(dict(G.degree()))
print("Adjacency matrix of graph G:")
print(nx.adjacency_matrix(G).todense())

# Draw a cycle graph
nx.draw(G, with_labels=True)
plt.show()

# Example 5: Finding the Shortest Path
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
shortest_path = nx.shortest_path(G, source=1, target=5)

print("Shortest path from node 1 to node 5 in graph G:")
print(shortest_path)

# Draw the graph with shortest path highlighted
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
path_edges = list(zip(shortest_path, shortest_path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)
plt.show()

# Example 6: Clustering Coefficient
G = nx.complete_graph(5)  # Creates a complete graph with 5 nodes
clustering_coeff = nx.clustering(G)
print("Clustering coefficient of each node in graph G:")
print(clustering_coeff)

# Draw a complete graph
nx.draw(G, with_labels=True)
plt.show()

# Create a directed graph to represent the flowlines of waterways
W = nx.DiGraph()

# Add nodes representing points along the waterways
# In a real example, these could be coordinates, labeled points, etc.
W.add_nodes_from(["Source", "A", "B", "C", "D", "E", "Sink"])

# Add edges representing the flowlines between points
# The weights can represent attributes like flow capacity, distance, etc.
W.add_edge("Source", "A", weight=5)
W.add_edge("Source", "B", weight=3)
W.add_edge("A", "C", weight=4)
W.add_edge("B", "C", weight=6)
W.add_edge("C", "D", weight=2)
W.add_edge("C", "E", weight=8)
W.add_edge("D", "Sink", weight=7)
W.add_edge("E", "Sink", weight=4)

# Print nodes and edges with attributes to verify the graph structure
print("Nodes of waterway graph W:")
print(W.nodes(data=True))
print("Edges of waterway graph W:")
print(W.edges(data=True))

# Draw the waterway graph
pos = nx.spectral_layout(W)  # Layout for better visual representation
nx.draw(W, pos, with_labels=True, node_color='lightblue', arrows=True)
edge_labels = nx.get_edge_attributes(W, 'weight')
nx.draw_networkx_edge_labels(W, pos, edge_labels=edge_labels)
plt.show()

# Example: Finding the shortest path from Source to Sink
shortest_path = nx.shortest_path(W, source="Source", target="Sink", weight='weight')
print("Shortest path from Source to Sink in waterway graph W:")
print(shortest_path)

# Draw the waterway graph with shortest path highlighted
nx.draw(W, pos, with_labels=True, node_color='lightblue', arrows=True)
nx.draw_networkx_edge_labels(W, pos, edge_labels=edge_labels)
path_edges = list(zip(shortest_path, shortest_path[1:]))
nx.draw_networkx_edges(W, pos, edgelist=path_edges, edge_color='r', width=2)
plt.show()