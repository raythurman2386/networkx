import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Dict, Tuple
from shapely.geometry import Point
from plotly.offline import plot
import plotly.graph_objs as go

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


def determine_flow_direction(row) -> Tuple[Point, Point]:
    """
    Determines the upstream and downstream points based on the FLOWDIR field.

    Args:
        row: GeoDataFrame row containing flowline information

    Returns:
        Tuple of (upstream_point, downstream_point)
    """
    line = row.geometry
    flowdir = row['flowdir']

    start_point = Point(line.coords[0])
    end_point = Point(line.coords[-1])

    # With Flow = 1, Against Flow = 2, Uninitialized = 0
    if flowdir == 1:
        return (start_point, end_point)
    elif flowdir == 2:
        return (end_point, start_point)
    else:
        # Default to start to end if flow direction is unknown
        return (start_point, end_point)

def create_nhd_graph(shapefile_path: str, tolerance: float = 1) -> nx.DiGraph:
    """
    Creates a directed graph from NHD Flowline data.

    Args:
        shapefile_path: Path to the NHD Flowline shapefile
        tolerance: Distance tolerance for connecting nodes (in the same units as the shapefile)

    Returns:
        NetworkX DiGraph representing the stream network
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Create directed graph
    G = nx.DiGraph()

    # Store node locations and metadata
    node_locations: Dict[str, Point] = {}

    # First pass: Create nodes and store their locations
    for idx, row in gdf.iterrows():
        upstream_point, downstream_point = determine_flow_direction(row)

        # Create unique IDs for upstream and downstream nodes
        upstream_id = f"node_{idx}_up"
        downstream_id = f"node_{idx}_down"

        # Store node locations
        node_locations[upstream_id] = upstream_point
        node_locations[downstream_id] = downstream_point

        # Add edge with attributes
        G.add_edge(
            upstream_id,
            downstream_id,
            reachcode=row['reachcode'],
            length=row['lengthkm'],
            feature_type=row['ftype'],
            geometry=row.geometry
        )

    # Second pass: Connect nodes that are within tolerance
    nodes = list(node_locations.items())
    for i, (node1_id, point1) in enumerate(nodes):
        for node2_id, point2 in nodes[i+1:]:
            if point1.distance(point2) <= tolerance:
                # Avoid self-loops and ensure proper flow direction
                if node1_id.endswith('_down') and node2_id.endswith('_up'):
                    G.add_edge(node1_id, node2_id, connection_type='confluence')
                elif node1_id.endswith('_up') and node2_id.endswith('_down'):
                    G.add_edge(node2_id, node1_id, connection_type='confluence')

    return G

def analyze_nhd_network(G: nx.DiGraph):
    """
    Analyzes and visualizes the NHD network.

    Args:
        G: NetworkX DiGraph representing the stream network
    """
    print(f"Network Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    # Identify main stem (longest path)
    try:
        longest_path = max(nx.all_simple_paths(G,
                                               source=list(G.nodes())[0],
                                               target=list(G.nodes())[-1]),
                           key=len)
        print(f"Length of longest path: {len(longest_path)}")
    except:
        print("Could not determine longest path")

    # Visualize the network
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw the network
    nx.draw(G, pos,
            node_size=20,
            node_color='blue',
            edge_color='gray',
            with_labels=False,
            arrows=True,
            arrowsize=10)

    plt.title("NHD Flowlines Network")
    plt.show()

def example_nhd():
    """Example usage with NHD data"""
    file_path = "NHDFlowline.shp"
    nhd_graph = create_nhd_graph(file_path)
    analyze_nhd_network(nhd_graph)


def minimum_spanning_tree_example():
    """Minimum Spanning Tree: Kruskal's Algorithm Example"""
    # Create a weighted undirected graph
    G = nx.Graph()
    edges = [
        ('A', 'B', 4), ('A', 'H', 8),
        ('B', 'C', 8), ('B', 'H', 11),
        ('C', 'D', 7), ('C', 'F', 4), ('C', 'I', 2),
        ('D', 'E', 9), ('D', 'F', 14),
        ('E', 'F', 10),
        ('F', 'G', 2),
        ('G', 'H', 1), ('G', 'I', 6),
        ('H', 'I', 7),
    ]
    G.add_weighted_edges_from(edges)

    # Compute the Minimum Spanning Tree using Kruskal's algorithm
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')

    # Print the edges in the MST
    print("Edges in the Minimum Spanning Tree:")
    for u, v, weight in mst.edges(data='weight'):
        print(f"({u}, {v}, {weight})")

    # Visualize the original graph and the MST
    pos = nx.spring_layout(G)

    plt.figure(figsize=(12, 6))

    # Original Graph
    plt.subplot(121)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Original Graph")

    # Minimum Spanning Tree
    plt.subplot(122)
    nx.draw(mst, pos, with_labels=True, node_color='lightgreen', edge_color='red')
    labels = nx.get_edge_attributes(mst, 'weight')
    nx.draw_networkx_edge_labels(mst, pos, edge_labels=labels)
    plt.title("Minimum Spanning Tree")

    plt.show()


def interactive_graph_visualization():
    """Interactive Graph Visualization with Plotly"""
    # Create a star graph
    G = nx.star_graph(n=10)

    # Compute positions for all nodes
    pos = nx.spring_layout(G)

    # Create edge traces
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    # Create node traces
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)

    for node, adjacents in G.adjacency():
        node_trace['marker']['color'] += (len(adjacents),)
        node_info = f'# of connections: {len(adjacents)}'
        node_trace['text'] += (node_info,)

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Interactive Network Graph',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Plotly NetworkX Integration",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # Display the plot
    plot(fig)


def dijkstra_shortest_path_example():
    """Dijkstra's Algorithm: Shortest Paths in Weighted Graph"""
    # Create a weighted directed graph
    G = nx.DiGraph()
    edges = [
        ('A', 'B', 1), ('A', 'C', 4),
        ('B', 'C', 2), ('B', 'D', 5),
        ('C', 'D', 1)
    ]
    G.add_weighted_edges_from(edges)

    # Compute the shortest path from 'A' to 'D' using Dijkstra's algorithm
    path = nx.dijkstra_path(G, source='A', target='D', weight='weight')
    print(f"Shortest path from 'A' to 'D': {path}")

    # Visualize the graph
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Highlight the shortest path
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    plt.title("Dijkstra's Shortest Path")
    plt.show()


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
        9: example_nhd,
        10: minimum_spanning_tree_example,
        11: interactive_graph_visualization,
        12: dijkstra_shortest_path_example,
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