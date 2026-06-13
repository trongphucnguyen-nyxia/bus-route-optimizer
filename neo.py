import os
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings('ignore')

# =========================================================
# SETTINGS
# =========================================================
FILES = {
    "routes": "busroute.geojson.geojsonl",
    "stops": "stops.geojson.geojsonl",
    "population": "population.geojson.geojsonl",
    "roads": "roads.geojson.geojsonl"
}

TARGET_CRS = "EPSG:3857"
WALK_BUFFER = 800  # 400m is ~5 min walk. Increase to 800 for "higher" coverage scores.
ITERATIONS = 100    # Increased iterations

class FortWorthOptimizer:
    def __init__(self):
        print("--- 1. Loading Fort Worth Data ---")
        self.routes = gpd.read_file(FILES["routes"], driver="GeoJSONSeq").to_crs(TARGET_CRS)
        self.stops = gpd.read_file(FILES["stops"], driver="GeoJSONSeq").to_crs(TARGET_CRS)
        self.pop_gdf = gpd.read_file(FILES["population"], driver="GeoJSONSeq").to_crs(TARGET_CRS)
        
        # Define bounds to keep the map focused
        self.bounds = self.pop_gdf.total_bounds 

        # Fix Route IDs
        if 'route_id' not in self.routes.columns:
            self.routes['route_id'] = self.routes.index.astype(str)
        self.routes['route_id'] = self.routes['route_id'].fillna("base").astype(str)

        # Fix Population Column
        pop_col = next((c for c in ['population','pop','DN','VALUE','pop_val'] if c in self.pop_gdf.columns), None)
        if pop_col:
            self.pop_gdf['pop_val'] = pd.to_numeric(self.pop_gdf[pop_col], errors='coerce').fillna(0)
        else:
            self.pop_gdf['pop_val'] = self.pop_gdf.geometry.area / 1000
        
        self.total_pop = self.pop_gdf['pop_val'].sum()
        
        # Pre-calculate centroids for fast coverage calculation
        self.pop_centroids = self.pop_gdf.copy()
        self.pop_centroids.geometry = self.pop_gdf.centroid

        print(f"--- 2. Building Road Graph ({FILES['roads']}) ---")
        roads_all = gpd.read_file(FILES["roads"], driver="GeoJSONSeq").to_crs(TARGET_CRS)
        
        # Crop roads to study area to keep the graph clean
        study_area = self.pop_gdf.geometry.union_all().envelope.buffer(2000)
        roads_gdf = roads_all[roads_all.geometry.intersects(study_area)]

        self.road_graph = nx.Graph()
        for geom in roads_gdf.geometry:
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                for i in range(len(coords) - 1):
                    u, v = coords[i], coords[i+1]
                    dist = Point(u).distance(Point(v))
                    self.road_graph.add_edge(u, v, weight=dist)
        
        self.nodes_list = list(self.road_graph.nodes)
        self.tree = cKDTree(self.nodes_list)
        print(f"Graph Ready: {len(self.nodes_list)} intersections.")

    def snap_to_road(self, point):
        dist, idx = self.tree.query([point.x, point.y])
        return self.nodes_list[idx]

    def calculate_coverage(self, current_stops):
        """Calculates percentage of population within WALK_BUFFER of any stop."""
        if current_stops.empty: return 0
        # Use spatial join for speed
        stop_buffer = current_stops.geometry.buffer(WALK_BUFFER).union_all()
        # Create a single geometry GDF for sjoin
        buffer_gdf = gpd.GeoDataFrame(geometry=[stop_buffer], crs=TARGET_CRS)
        served = gpd.sjoin(self.pop_centroids, buffer_gdf, how='inner', predicate='within')
        return (served['pop_val'].sum() / self.total_pop) * 100

    def propose_route(self, current_stops, step):
        """Finds the best underserved area and creates a path to the nearest existing stop."""
        # Find areas not covered
        buffer = current_stops.geometry.buffer(WALK_BUFFER).union_all()
        underserved = self.pop_gdf[~self.pop_gdf.geometry.intersects(buffer)].copy()
        
        if underserved.empty: 
            return None, None

        # Sort by population - check top 10 candidates if pathfinding fails
        candidates = underserved.sort_values(by='pop_val', ascending=False).head(10)
        
        stop_coords = [(p.x, p.y) for p in current_stops.geometry]
        stop_tree = cKDTree(stop_coords)

        for _, candidate in candidates.iterrows():
            target_pt = candidate.geometry.centroid
            start_node = self.snap_to_road(target_pt)
            
            # Find closest existing stop to connect to
            _, stop_idx = stop_tree.query([target_pt.x, target_pt.y])
            nearest_stop_pt = current_stops.iloc[stop_idx].geometry
            end_node = self.snap_to_road(nearest_stop_pt)

            # Only proceed if a path exists in the road graph
            if nx.has_path(self.road_graph, start_node, end_node):
                try:
                    path_coords = nx.shortest_path(self.road_graph, start_node, end_node, weight='weight')
                    if len(path_coords) < 2: continue
                    
                    route_line = LineString(path_coords)
                    rid = f"NEW_{step}_{np.random.randint(100,999)}"
                    
                    new_route_gdf = gpd.GeoDataFrame([{'route_id': rid, 'geometry': route_line}], crs=TARGET_CRS)
                    
                    # Create stops every 400m
                    new_stops_list = []
                    for d in np.arange(0, route_line.length, 400):
                        new_stops_list.append({'route_id': rid, 'geometry': route_line.interpolate(d)})
                    new_stops_list.append({'route_id': rid, 'geometry': route_line.interpolate(route_line.length)})
                    
                    return new_route_gdf, gpd.GeoDataFrame(new_stops_list, crs=TARGET_CRS)
                except:
                    continue
        
        return None, None

    def run(self):
        curr_routes = self.routes.copy()
        curr_stops = self.stops.copy()
        
        print(f"--- 3. Running {ITERATIONS} Iterations ---")
        
        for i in range(ITERATIONS + 1):
            # Calculate Coverage
            cov = self.calculate_coverage(curr_stops)
            print(f"Step {i}/{ITERATIONS} | Coverage: {cov:.2f}%")

            # Save Visual (Only every 5 steps to save time)
            if i % 5 == 0 or i == ITERATIONS:
                fig, ax = plt.subplots(figsize=(12, 12))
                self.pop_gdf.plot(ax=ax, color='#f2f2f2', edgecolor='#d1d1d1', alpha=0.5)
                
                # Plot existing network
                curr_routes[~curr_routes['route_id'].str.contains("NEW", na=False)].plot(
                    ax=ax, color='blue', linewidth=0.5, alpha=0.4, label='Original'
                )
                
                # Plot new network
                new_mask = curr_routes['route_id'].str.contains("NEW", na=False)
                if new_mask.any():
                    curr_routes[new_mask].plot(
                        ax=ax, color='red', linewidth=1.5, label='Optimized'
                    )
                
                ax.set_xlim([self.bounds[0], self.bounds[2]])
                ax.set_ylim([self.bounds[1], self.bounds[3]])
                plt.title(f"Fort Worth Transit Optimization - Step {i}\nCoverage: {cov:.2f}%", fontsize=14)
                plt.savefig(f"step_{i}.png", dpi=150, bbox_inches='tight')
                plt.close()

            # Propose next route for next iteration
            if i < ITERATIONS:
                new_r, new_s = self.propose_route(curr_stops, i)
                if new_r is not None:
                    curr_routes = pd.concat([curr_routes, new_r], ignore_index=True)
                    curr_stops = pd.concat([curr_stops, new_s], ignore_index=True)
                else:
                    print("No more valid paths found. Stopping early.")
                    break

        # Save Final Result
        curr_routes.to_file("optimized_fort_worth_network.geojson", driver="GeoJSON")
        print(f"Success. Final coverage: {cov:.2f}%. Files saved.")

if __name__ == "__main__":
    FortWorthOptimizer().run()
