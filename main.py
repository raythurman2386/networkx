import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point


def basic_graph_operations():
    """Basic Graph Operations: Adding and Removing Nodes and Edges"""
    H = nx.Graph()
    H.add_nodes_from([1, 2, 3, 4, 5])
    H.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])

    print("Nodes of graph H:", H.nodes())
    print("Edges of graph H:", H.edges())

    H.remove_node(5)
    H.remove_edge(1, 2)

    nx.draw(H, with_labels=True)
    plt.title("Basic Graph Operations: Modified Graph")
    plt.show()


def directed_graph_example():
    """Directed Graph: Creating and Visualizing"""
    D = nx.DiGraph()
    D.add_edges_from([(1, 2), (2, 3), (3, 1)])

    print("Nodes of directed graph D:", D.nodes())
    print("Edges of directed graph D:", D.edges())

    nx.draw(D, with_labels=True, node_color='lightblue', arrows=True)
    plt.title("Directed Graph Example")
    plt.show()


def graph_attributes():
    """Graph Attributes: Adding and Accessing Node and Edge Attributes"""
    G = nx.Graph()
    G.add_node(1, label='A')
    G.add_node(2, label='B')
    G.add_edge(1, 2, weight=4.2)

    print("Node attributes in graph G:", G.nodes(data=True))
    print("Edge attributes in graph G:", G.edges(data=True))


def graph_analysis():
    """Graph Analysis: Degree and Adjacency Matrix"""
    G = nx.cycle_graph(4)
    print("Degree of each node in graph G:", dict(G.degree()))
    print("Adjacency matrix of graph G:\n", nx.adjacency_matrix(G).todense())

    nx.draw(G, with_labels=True)
    plt.title("Graph Analysis: Cycle Graph")
    plt.show()


def shortest_path_visualization():
    """Shortest Path: Finding and Visualizing"""
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
    shortest_path = nx.shortest_path(G, source=1, target=5)

    print("Shortest path from node 1 to node 5 in graph G:", shortest_path)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    path_edges = list(zip(shortest_path, shortest_path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)
    plt.title("Shortest Path Visualization")
    plt.show()


def clustering_coefficient():
    """Clustering Coefficient: Complete Graph Example"""
    G = nx.complete_graph(5)
    clustering_coeff = nx.clustering(G)
    print("Clustering coefficient of each node in graph G:", clustering_coeff)

    nx.draw(G, with_labels=True)
    plt.title("Clustering Coefficient: Complete Graph")
    plt.show()


def example_waterway():
    """Waterway Flowlines Graph"""
    W = nx.DiGraph()
    W.add_nodes_from(["Source", "A", "B", "C", "D", "E", "Sink"])
    W.add_edge("Source", "A", weight=5)
    W.add_edge("Source", "B", weight=3)
    W.add_edge("A", "C", weight=4)
    W.add_edge("B", "C", weight=6)
    W.add_edge("C", "D", weight=2)
    W.add_edge("C", "E", weight=8)
    W.add_edge("D", "Sink", weight=7)
    W.add_edge("E", "Sink", weight=4)

    print("Nodes of waterway graph W:", W.nodes(data=True))
    print("Edges of waterway graph W:", W.edges(data=True))

    pos = nx.spectral_layout(W)
    nx.draw(W, pos, with_labels=True, node_color='lightblue', arrows=True)
    edge_labels = nx.get_edge_attributes(W, 'weight')
    nx.draw_networkx_edge_labels(W, pos, edge_labels=edge_labels)
    plt.title("Waterway Flowlines Graph")
    plt.show()

    shortest_path = nx.shortest_path(W, source="Source", target="Sink", weight='weight')
    print("Shortest path from Source to Sink in waterway graph W:", shortest_path)

    nx.draw(W, pos, with_labels=True, node_color='lightblue', arrows=True)
    nx.draw_networkx_edge_labels(W, pos, edge_labels=edge_labels)
    path_edges = list(zip(shortest_path, shortest_path[1:]))
    nx.draw_networkx_edges(W, pos, edgelist=path_edges, edge_color='r', width=2)
    plt.title("Shortest Path in Waterway Flowlines Graph")
    plt.show()


def example_river_system():
    """River System Flowlines Graph"""
    R = nx.DiGraph()
    R.add_nodes_from(["Source", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "Sink"])
    edges = [
        ("Source", "A", 10), ("Source", "B", 15), ("A", "C", 7), ("A", "D", 3),
        ("B", "D", 10), ("B", "E", 5), ("C", "F", 7), ("D", "G", 8),
        ("E", "G", 6), ("E", "H", 4), ("F", "I", 5), ("G", "J", 12),
        ("H", "J", 4), ("I", "K", 2), ("J", "L", 9), ("K", "L", 3),
        ("L", "Sink", 14)
    ]
    R.add_weighted_edges_from(edges, weight='flow_rate')

    print("Nodes of river system graph R:", R.nodes(data=True))
    print("Edges with flow rates in river system graph R:", R.edges(data=True))

    pos = nx.spring_layout(R)
    nx.draw(R, pos, with_labels=True, node_color='lightblue', arrows=True)
    edge_labels = nx.get_edge_attributes(R, 'flow_rate')
    nx.draw_networkx_edge_labels(R, pos, edge_labels=edge_labels)
    plt.title("River System Flowlines Graph")
    plt.show()

    shortest_path = nx.shortest_path(R, source="Source", target="Sink", weight='flow_rate')
    print("Shortest path from Source to Sink in river system graph R by flow rate:", shortest_path)

    nx.draw(R, pos, with_labels=True, node_color='lightblue', arrows=True)
    nx.draw_networkx_edge_labels(R, pos, edge_labels=edge_labels)
    path_edges = list(zip(shortest_path, shortest_path[1:]))
    nx.draw_networkx_edges(R, pos, edgelist=path_edges, edge_color='r', width=2)
    plt.title("Shortest Path in River System Flowlines Graph by Flow Rate")
    plt.show()

    flow_value, flow_dict = nx.maximum_flow(R, "Source", "Sink", capacity='flow_rate')
    print("Maximum flow from Source to Sink in river system graph R:", flow_value)

    edge_colors = ['blue' if edge in path_edges else 'black' for edge in R.edges()]
    nx.draw(R, pos, with_labels=True, node_color='lightblue', arrows=True, edge_color=edge_colors)
    nx.draw_networkx_edge_labels(R, pos, edge_labels=edge_labels)
    plt.title("Maximum Flow in River System Flowlines Graph")
    plt.show()


def create_nhd_graph(shapefile_path, tolerance=1e-6):
    gdf = gpd.read_file(shapefile_path)
    G = nx.DiGraph()
    start_points, end_points = {}, {}

    for idx, row in gdf.iterrows():
        line = row.geometry
        start_points[idx] = Point(line.coords[0])
        end_points[idx] = Point(line.coords[-1])

    def find_nearest(point, points_dict):
        return next((idx for idx, p in points_dict.items() if point.distance(p) <= tolerance), None)

    for idx, row in gdf.iterrows():
        line = row.geometry
        start_node = find_nearest(Point(line.coords[0]), end_points) or idx
        end_node = find_nearest(Point(line.coords[-1]), start_points) or f"{idx}_end"
        G.add_edge(start_node, end_node)

    return G


def example_nhd():
    """National Hydrography Dataset Example"""
    file_path = "NHDFlowline.shp"
    nhd_graph = create_nhd_graph(file_path)

    print(f"Number of nodes: {nhd_graph.number_of_nodes()}")
    print(f"Number of edges: {nhd_graph.number_of_edges()}")

    pos = nx.spring_layout(nhd_graph)
    plt.figure(figsize=(12, 8))
    nx.draw(nhd_graph, pos, node_size=20, node_color='blue', with_labels=False, arrows=True)
    plt.title("NHD Flowlines Network")
    plt.show()

    connected_components = list(nx.weakly_connected_components(nhd_graph))
    print(f"Number of weakly connected components: {len(connected_components)}")
    print(f"Size of the largest component: {len(max(connected_components, key=len))}")

    centrality = nx.degree_centrality(nhd_graph)
    top_5_central_nodes = sorted(centrality, key=centrality.get, reverse=True)[:5]
    print("Top 5 nodes by degree centrality:")
    for node in top_5_central_nodes:
        print(f"Node: {node}, Centrality: {centrality[node]}")


def main():
    examples = {
        1: basic_graph_operations,
        2: directed_graph_example,
        3: graph_attributes,
        4: graph_analysis,
        5: shortest_path_visualization,
        6: clustering_coefficient,
        7: example_waterway,
        8: example_river_system,
        9: example_nhd
    }

    while True:
        print("\nAvailable examples:")
        for num, func in examples.items():
            print(f"{num}. {func.__doc__}")
        print("0. Exit")

        choice = input("Enter the number of the example you want to run (0 to exit): ")
        if choice == '0':
            break
        elif choice.isdigit() and int(choice) in examples:
            examples[int(choice)]()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()