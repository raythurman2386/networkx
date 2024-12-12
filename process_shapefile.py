import json
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import split, snap, nearest_points
import networkx as nx
from typing import Union, List, Dict, Optional
import os
from datetime import datetime
import logging
from dataclasses import dataclass

@dataclass
class FlowlineMetrics:
    """Data class to store flowline analysis metrics"""
    total_length: float
    segment_count: int
    density: float
    sinuosity: float
    connectivity_index: float


class FlowlineAnalyzer:
    """
    A comprehensive class for analyzing stream networks and flowlines
    """
    def __init__(self, flowline_path: str, output_dir: str = None):
        """
        Initialize the FlowlineAnalyzer with input data and setup

        Parameters:
        -----------
        flowline_path : str
            Path to the input flowline shapefile
        output_dir : str, optional
            Directory for output files
        """
        self.logger = self._setup_logging()
        self.flowline_path = flowline_path
        self.output_dir = self._setup_output_directory(output_dir)
        self.gdf = self._load_data()
        self.graph = None
        self.points_gdf = None
        self.metrics = None

    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger('FlowlineAnalyzer')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    def _setup_output_directory(self, output_dir: str = None) -> str:
        """Setup output directory structure with overwrite capability"""
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'flowline_analysis')

        # Create or clean existing directory
        if os.path.exists(output_dir):
            self.logger.info(f"Overwriting existing directory: {output_dir}")
            for subdir in ['points', 'networks', 'metrics', 'styles']:
                subdir_path = os.path.join(output_dir, subdir)
                if os.path.exists(subdir_path):
                    for file in os.listdir(subdir_path):
                        file_path = os.path.join(subdir_path, file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            self.logger.warning(f"Error removing file {file_path}: {e}")

        # Create directories
        for subdir in ['points', 'networks', 'metrics', 'styles']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        return output_dir

    def _load_data(self) -> gpd.GeoDataFrame:
        """Load and validate input data with proper geometry handling"""
        try:
            # Read the data
            gdf = gpd.read_file(self.flowline_path)

            # Handle M-geometries by converting to 3D LineString
            if any(gdf.geometry.has_z):
                gdf.geometry = gdf.geometry.apply(lambda geom: LineString(
                    [(x, y, z) for x, y, z, *_ in geom.coords]
                ))

            # Ensure CRS is set and project to a suitable coordinate system
            if gdf.crs is None:
                self.logger.warning("Input data has no CRS. Attempting to set default (EPSG:4326)")
                gdf.set_crs(epsg=4326, inplace=True)

            # Project to a suitable coordinate system for accurate measurements
            # Using USA Contiguous Albers Equal Area projection
            gdf = gdf.to_crs(epsg=5070)

            self.logger.info(f"Loaded {len(gdf)} flowline features")
            self.logger.info(f"CRS: {gdf.crs}")
            self.logger.info(f"Geometry type: {gdf.geometry.geom_type.unique()}")

            return gdf

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _create_qgis_style(self, layer_type: str) -> str:
        """
        Create QGIS style files for different layer types

        Parameters:
        -----------
        layer_type : str
            Type of layer ('points', 'stream_order', 'network')
        """
        style_content = ""

        if layer_type == 'points':
            style_content = """
<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.0" styleCategories="Symbology">
  <renderer-v2 type="graduatedSymbol" attr="PERCENT">
    <ranges>
      <range lower="0.000000" upper="25.000000" symbol="0" label="0-25%"/>
      <range lower="25.000000" upper="50.000000" symbol="1" label="25-50%"/>
      <range lower="50.000000" upper="75.000000" symbol="2" label="50-75%"/>
      <range lower="75.000000" upper="100.000000" symbol="3" label="75-100%"/>
    </ranges>
    <symbols>
      <symbol type="marker" name="0"><prop k="color" v="68,1,84,255"/></symbol>
      <symbol type="marker" name="1"><prop k="color" v="59,82,139,255"/></symbol>
      <symbol type="marker" name="2"><prop k="color" v="33,144,141,255"/></symbol>
      <symbol type="marker" name="3"><prop k="color" v="53,183,121,255"/></symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""
        elif layer_type == 'stream_order':
            style_content = """
<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.0" styleCategories="Symbology">
  <renderer-v2 type="graduatedSymbol" attr="stream_order">
    <ranges>
      <range lower="1" upper="1" symbol="0" label="Order 1"/>
      <range lower="2" upper="2" symbol="1" label="Order 2"/>
      <range lower="3" upper="3" symbol="2" label="Order 3"/>
      <range lower="4" upper="4" symbol="3" label="Order 4"/>
      <range lower="5" upper="5" symbol="4" label="Order 5+"/>
    </ranges>
    <symbols>
      <symbol type="line" name="0"><prop k="line_width" v="0.5"/><prop k="color" v="227,227,227,255"/></symbol>
      <symbol type="line" name="1"><prop k="line_width" v="1.0"/><prop k="color" v="189,189,189,255"/></symbol>
      <symbol type="line" name="2"><prop k="line_width" v="1.5"/><prop k="color" v="150,150,150,255"/></symbol>
      <symbol type="line" name="3"><prop k="line_width" v="2.0"/><prop k="color" v="99,99,99,255"/></symbol>
      <symbol type="line" name="4"><prop k="line_width" v="2.5"/><prop k="color" v="0,0,0,255"/></symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""

        style_path = os.path.join(self.output_dir, 'styles', f'{layer_type}.qml')
        with open(style_path, 'w', encoding='utf-8') as f:
            f.write(style_content)
        return style_path

    def create_points_along_lines(self, spacing: float = 100) -> gpd.GeoDataFrame:
        """Create points along flowlines with specified spacing"""
        try:
            points = []
            attributes = []

            for idx, row in self.gdf.iterrows():
                line = row.geometry
                if line is None or line.is_empty:
                    continue

                line_length = line.length
                num_points = max(2, int(line_length / spacing))
                distances = np.linspace(0, line_length, num_points)

                for distance in distances:
                    point = line.interpolate(distance)
                    percent_along = (distance / line_length) * 100

                    points.append(point)
                    attributes.append({
                        'ORIG_FID': row.name,
                        'DISTANCE': round(distance, 2),
                        'PERCENT': round(percent_along, 2),
                        **{col: row[col] for col in self.gdf.columns if col != 'geometry'}
                    })

            self.points_gdf = gpd.GeoDataFrame(
                attributes,
                geometry=points,
                crs=self.gdf.crs
            )

            return self.points_gdf

        except Exception as e:
            self.logger.error(f"Error creating points: {str(e)}")
            raise

    def create_network_graph(self) -> nx.DiGraph:
        """Convert flowlines to a NetworkX directed graph for network analysis"""
        try:
            G = nx.DiGraph()

            for idx, row in self.gdf.iterrows():
                line = row.geometry
                if line is None or line.is_empty:
                    continue

                # Extract start and end points
                start_point = Point(line.coords[0])
                end_point = Point(line.coords[-1])

                # Add edge to graph
                G.add_edge(
                    str(start_point.coords[0]),
                    str(end_point.coords[0]),
                    geometry=line,
                    length=line.length,
                    **{col: row[col] for col in self.gdf.columns if col != 'geometry'}
                )

            self.graph = G
            self.logger.info(f"Created network graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G

        except Exception as e:
            self.logger.error(f"Error creating network graph: {str(e)}")
            raise

    def _calculate_network_statistics(self) -> Dict:
        """Calculate detailed network statistics"""
        try:
            stats = {
                'total_length': self.gdf.geometry.length.sum(),
                'mean_length': self.gdf.geometry.length.mean(),
                'std_length': self.gdf.geometry.length.std(),
                'segment_count': len(self.gdf),
                'node_count': self.graph.number_of_nodes() if self.graph else 0,
                'edge_count': self.graph.number_of_edges() if self.graph else 0,
            }
            return stats
        except Exception as e:
            self.logger.error(f"Error calculating network statistics: {str(e)}")
            raise

    def _calculate_stream_order(self) -> gpd.GeoDataFrame:
        """Calculate Strahler stream order"""
        try:
            if self.graph is None:
                self.create_network_graph()

            # Initialize all streams as order 1
            nx.set_edge_attributes(self.graph, 1, 'stream_order')

            # Find outlet nodes (nodes with only incoming edges)
            outlets = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]

            # Process from outlets upstream
            for outlet in outlets:
                self._process_upstream_order(outlet)

            # Create GeoDataFrame with stream orders
            stream_orders = []
            for u, v, data in self.graph.edges(data=True):
                stream_orders.append({
                    'geometry': data['geometry'],
                    'stream_order': data['stream_order'],
                    'length': data['length']
                })

            return gpd.GeoDataFrame(stream_orders, crs=self.gdf.crs)

        except Exception as e:
            self.logger.error(f"Error calculating stream order: {str(e)}")
            raise

    def _process_upstream_order(self, node: str):
        """Helper function for stream order calculation"""
        incoming_edges = list(self.graph.in_edges(node, data=True))

        if not incoming_edges:
            return 1

        # Get orders of incoming streams
        incoming_orders = [self._process_upstream_order(u) for u, v, _ in incoming_edges]

        # Calculate new order based on Strahler method
        if len(incoming_orders) == 1:
            new_order = incoming_orders[0]
        else:
            max_order = max(incoming_orders)
            if incoming_orders.count(max_order) > 1:
                new_order = max_order + 1
            else:
                new_order = max_order

        # Set order for all incoming edges
        for u, v, data in incoming_edges:
            self.graph[u][v]['stream_order'] = new_order

        return new_order

    def save_results(self):
        """Save all analysis results as QGIS-compatible layers"""
        try:
            # Save points layer
            if self.points_gdf is not None:
                points_path = os.path.join(self.output_dir, 'points', 'flowline_points.gpkg')
                if self.points_gdf.crs is None:
                    self.points_gdf.set_crs(self.gdf.crs, inplace=True)
                self.points_gdf.to_file(points_path, driver='GPKG')
                self._create_qgis_style('points')
                self.logger.info(f"Saved points layer to {points_path}")

            # Save stream order layer
            stream_order_gdf = self._calculate_stream_order()
            order_path = os.path.join(self.output_dir, 'networks', 'stream_order.gpkg')
            stream_order_gdf.to_file(order_path, driver='GPKG')
            self._create_qgis_style('stream_order')
            self.logger.info(f"Saved stream order layer to {order_path}")

            # Save network statistics as GeoJSON with attributes
            stats = self._calculate_network_statistics()
            network_gdf = self.gdf.copy()
            for key, value in stats.items():
                network_gdf[key] = value
            network_path = os.path.join(self.output_dir, 'networks', 'network_stats.gpkg')
            network_gdf.to_file(network_path, driver='GPKG')
            self.logger.info(f"Saved network statistics layer to {network_path}")

            # Save statistics as JSON
            stats_path = os.path.join(self.output_dir, 'metrics', 'network_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4, default=str)

            self.logger.info(f"Results saved to {self.output_dir}")

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

    def load_in_qgis(self):
        """Load all layers into QGIS with proper styling"""
        try:
            from qgis.utils import iface
            from qgis.core import QgsVectorLayer, QgsProject

            # Create a group for our layers
            root = QgsProject.instance().layerTreeRoot()
            group = root.addGroup("Flowline Analysis")

            # Add layers with styling
            layers = {
                'Points': os.path.join(self.output_dir, 'points', 'flowline_points.gpkg'),
                'Stream Order': os.path.join(self.output_dir, 'networks', 'stream_order.gpkg'),
                'Network Statistics': os.path.join(self.output_dir, 'networks', 'network_stats.gpkg')
            }

            for name, path in layers.items():
                layer = QgsVectorLayer(path, name, "ogr")
                if layer.isValid():
                    # Apply styling
                    style_path = os.path.join(self.output_dir, 'styles', f'{name.lower().replace(" ", "_")}.qml')
                    if os.path.exists(style_path):
                        layer.loadNamedStyle(style_path)

                    # Add to the group
                    QgsProject.instance().addMapLayer(layer, False)
                    group.addLayer(layer)

            # Zoom to layers
            iface.zoomToActiveLayer()

            self.logger.info("Layers loaded in QGIS")

        except ImportError:
            self.logger.warning("Not running in QGIS environment")
        except Exception as e:
            self.logger.error(f"Error loading layers in QGIS: {str(e)}")


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = FlowlineAnalyzer("nhdflowline.shp")

    # Create points
    points = analyzer.create_points_along_lines(spacing=100)

    # Create network graph
    graph = analyzer.create_network_graph()

    # Save all results
    analyzer.save_results()