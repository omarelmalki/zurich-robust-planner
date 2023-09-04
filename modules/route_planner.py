import datetime
import pandas as pd
import scipy.stats as stats
from database import Database
from collections import defaultdict
from collections import deque


class RoutePlanner:
    """
    Finds the shortest path between two nodes in a network graph.
    Each edge has departure and arrival times which means all edges are not compatible if their timings aren't compatible.
    Use Dijkstra's algorithm to find the shortest path.
    """
    def __init__(self, 
                 network_graph, 
                 source_node, 
                 target_node,
                 deadline,
                 database
                ):
        self.G = network_graph
        self.source = source_node
        self.target = target_node
        self.deadline = deadline
        
        assert self.source in set(self.G.nodes()), "The source stop is not present in our graph"
        assert self.target in set(self.G.nodes()), "The target stop is not present in our graph"
        
        self.delay_database = database
        self.day_periods = {
            "morning": ("06:00", "08:30"),
            "prenoon": ("08:30", "12:00"),
            "afternoon": ("12:00", "15:00"),
            "latenoon": ("15:00", "18:00"),
            "evening": ("18:00", "22:00")
        }
        
    def extract_path(self, node, previous_edge):
        """
        Extract path of edges to a node and the dictionary of previous_edges.
        """
        res = []
        # if node has a predecessor, add path, if not this means we have reached source node
        while node in previous_edge:
            res.append(previous_edge[node])
            node = previous_edge[node][0]
        return res
        
    def extract_trips(self, path):
        """
        Extract trip_ids taken by path
        """
        trips = set()
        for edge in path:
            trips.add(edge[2]['trip_id'])
        return trips
        
    def find_paths(self, n=1):
        """
        Output n paths to the user from the source to destination
        """
        n = max(1, n)
        res = []
        paths = []
        # initialise queue of banned trips
        banned_trips_queue = deque()
        banned_trips_queue.append(set())
        while n > 0 and banned_trips_queue:
            bt = banned_trips_queue.popleft() 
            # get shortest path omitting trips in bt
            node, previous_edge, dist, walk, n_transfers = self.dijkstra_algorithm(banned_trips=bt)
            # if trip found
            if node:
                n -= 1
                path = self.extract_path(node, previous_edge)
                # add elements in sorting priority (last dep. time, walk_time, n_transfers, index in paths array)
                res.append((dist, walk, n_transfers, len(paths)))
                paths.append(path)
                trips = self.extract_trips(path)
                for t in trips:
                    if t:
                        # add new trip combination to ban
                        banned_trips_queue.append(set(list(bt) + [t]))
                        
        # compute success probability
        res = sorted(res)
        routes_with_success_proba = []
        for route in res:
            routes_with_success_proba.append((paths[route[3]], self.compute_success_probability(paths[route[3]])))
        
        return routes_with_success_proba
    
    def compute_success_probability(self, route):
        """Computes the probability of success of the route.

        Returns:
            p_success (float): the probability of success of the route.
        """                        
        def transfer_success_probability(edge_data):
            
            assert edge_data is not None
            assert edge_data["next_departure_time"] is not None
            assert edge_data["next_arrival_time"] is not None
            
            # convert strings to datetime
            next_departure_time = pd.to_datetime(edge_data["next_departure_time"], format="%H:%M:%S")
            next_arrival_time = pd.to_datetime(edge_data["next_arrival_time"], format="%H:%M:%S")
            
            # determine day period
            formatted_arrival_time = next_arrival_time.strftime("%H:%M")
            if formatted_arrival_time < self.day_periods["morning"][1]:
                day_period = "morning"
            elif formatted_arrival_time < self.day_periods["prenoon"][1]:
                day_period = "prenoon"
            elif formatted_arrival_time < self.day_periods["afternoon"][1]:
                day_period = "afternoon"
            elif formatted_arrival_time < self.day_periods["latenoon"][1]:
                day_period = "latenoon"
            else:
                day_period = "evening"

            if not self.delay_database:
                delay_mean = None
                delay_std = None
            else:
                # compute mean and std of the delay for the arrival stop of the edge
                delay_mean = self.delay_database.fetch_delay(
                    metric="avg", stop_name=edge_data["stop_name"], day_period=day_period,
                    transport_type=edge_data["transport_type"], transport_subtype=edge_data["transport_subtype"]
                )
                delay_std = self.delay_database.fetch_delay(
                    metric="std", stop_name=edge_data["stop_name"], day_period=day_period,
                    transport_type=edge_data["transport_type"], transport_subtype=edge_data["transport_subtype"]
                )
            max_possible_delay_s = (next_departure_time - next_arrival_time).seconds
            
            # compute the probability of success of the transfer
            if delay_mean is None or delay_std is None:
                # we don't have an estimate for the given stop name
                assert delay_mean is None and delay_std is None, "either both mean and std are null or neither is."
                p_success = 1.0
            else:
                p_success = stats.norm.cdf(max_possible_delay_s, loc=delay_mean, scale=delay_std)
                
            return p_success
        
        # start at the first non-walkable edge
        current_trip_id = None
        i = 0
        for _, _, edge_data in route:
            current_trip_id = edge_data["trip_id"]
            if current_trip_id is not None:
                break
            i += 1
        
        # if there is no non-walkable edge, probability of success is 1
        if current_trip_id is None:
            # the route must contain only walkable edges
            for _, _, edge_data in route:
                assert edge_data["transport_type"] == "walk", f"{edge_data} must be a walkable edge since its trip_id is None."
            return 1.0

        # otherwise, the route has at least one non-walkable edge
        # compute route's probability of success 
        route_p_success = 1.0
        for _, _, edge_data in route[1:]:
            if edge_data["trip_id"] is None:
                # must be a walkable edge or else there is an issue
                assert edge_data["transport_type"] == "walk", f"{edge_data} must be a walkable edge since its trip_id is None."
                continue
            if (edge_data["next_departure_time"] is None) or (edge_data["next_arrival_time"] is None):
                # next edge is walkable, no way to check that aside from:
                assert (edge_data["next_departure_time"] is None) and (edge_data["next_arrival_time"] is None), "both next_departure_time and next_arrival_time must be None or neither is."
                continue
            # otherwise, if the edge is a transfer, compute its probability of success
            is_transfer = (current_trip_id != edge_data["trip_id"])
            if is_transfer:
                route_p_success *= transfer_success_probability(edge_data)
                current_trip_id = edge_data["trip_id"]
        
        return route_p_success
    
    def dijkstra_algorithm(self, transfer_time=120, banned_trips=set()):
        """
        Returns the shortest path between source and target nodes.
        """
        
        # initialize dictionaries for distance, walking_time, n_transfers and previous edge in optimal path
        distance = defaultdict(lambda: float('inf'))
        walking_time = defaultdict(lambda: float('inf'))
        number_of_transfers = defaultdict(lambda: float('inf'))
        distance[self.target] = 0 # distance(N) = deadline - departure_time from node N of shortest path to target
        walking_time[self.target] = 0
        number_of_transfers[self.target] = 0
        previous_edge = {}
        
        # set of edges not yet visited
        not_visited = set(self.G.nodes())
        not_visited.remove(self.target)
        
        # indicator of if a path was found diring durrent iteration
        path_found = False
        
        # indication of the distance of shortest path yet 
        distance_found = float('inf')

        edges = self.G.edges(self.target, data=True)

        for edge in edges:
            edge_data = edge[2]
            if edge_data['trip_id'] in banned_trips:
                # if edge in banned trips, skip
                continue
                
            if edge_data['is_walkable'] or (datetime.datetime.strptime(edge_data['next_arrival_time'], '%H:%M:%S') <= self.deadline):
                # add the edge to the path
                w = 0
                if edge_data['is_walkable'] == True:
                    # set departure time of walking edge for use by next steps of algorithm
                    edge_data['departure_time'] = (self.deadline - datetime.timedelta(seconds=int(edge_data['duration_s']))).strftime("%H:%M:%S")
                    w += edge_data['duration_s']
 
                d = (self.deadline - datetime.datetime.strptime(edge_data['departure_time'], '%H:%M:%S')).total_seconds()
    
                if edge[1] == self.source:
                    path_found = True
                    # do not count the walking time of the first edge, if both edges have the same stop_name
                    if edge_data['is_walkable'] and edge_data['stop_name'] == edge_data['next_stop_name']:
                        d = d - edge_data['duration_s']
                        w = w - edge_data['duration_s']
                
                if d < distance[edge[1]] or (d == distance[edge[1]] and w < walking_time[edge[1]]):
                    distance[edge[1]] = d
                    walking_time[edge[1]] = w
                    number_of_transfers[edge[1]] = 0
                    previous_edge[edge[1]] = edge
    
        if path_found:
            distance_found = distance[self.source]
        
        for i in range(self.G.number_of_nodes() - 1):
            u = min(not_visited, key=lambda x: distance[x])
            not_visited.remove(u)
            
            # if there is no current path (still not arrived at source, 
            # that has a departure time later or equal than the current shortest path found
            # this means we have found the shortest path and we stop
            if distance[u] > distance_found:
                break

            edges = self.G.edges(u, data=True)
            
            path_found = False
            
            for edge in edges:
                edge_data = edge[2]
                if edge_data['trip_id'] in banned_trips:
                    continue

                prev_edge = previous_edge[u]
                if not (edge_data['is_walkable'] and prev_edge[2]['is_walkable']):
                    # if we continue on the same trip or if the next edge is walkable, no need to check for timing compatibility
                    if edge_data['is_walkable'] or (edge_data['trip_id'] == prev_edge[2]['trip_id']):
                        w = 0
                        if edge_data['is_walkable'] == True:
                            edge[2]['departure_time'] = (datetime.datetime.strptime(prev_edge[2]['departure_time'],'%H:%M:%S') - datetime.timedelta(seconds=int(edge_data['duration_s']))).strftime("%H:%M:%S")
                            w += edge_data['duration_s']
    
                        d = (datetime.datetime.strptime(prev_edge[2]['departure_time'], '%H:%M:%S') - datetime.datetime.strptime(edge_data['departure_time'], '%H:%M:%S')).total_seconds()

                        if edge[1] == self.source:
                            path_found = True

                        new_d = distance[u] + d
                        new_w = walking_time[u] + w
                        new_n = number_of_transfers[u]

                        if new_d < distance[edge[1]] or (new_d == distance[edge[1]] and new_w < walking_time[edge[1]]) or (new_d == distance[edge[1]] and new_w == walking_time[edge[1]] and new_n < number_of_transfers[edge[1]]):
                            distance[edge[1]] = new_d
                            walking_time[edge[1]] = new_w
                            number_of_transfers[edge[1]] = new_n
                            previous_edge[edge[1]] = edge
                    else:
                        # verify that a transfer is possible
                        waiting_time = (datetime.datetime.strptime(prev_edge[2]['departure_time'], '%H:%M:%S') - datetime.timedelta(seconds=transfer_time)) - datetime.datetime.strptime(edge_data['next_arrival_time'], '%H:%M:%S')
                        waiting_seconds = waiting_time.total_seconds()
                        if waiting_seconds >= 0:
                            d = (datetime.datetime.strptime(prev_edge[2]['departure_time'], '%H:%M:%S') - datetime.datetime.strptime(edge_data['departure_time'], '%H:%M:%S')).total_seconds()

                            new_d = distance[u] + d
                            new_w = walking_time[u]
                            new_n = number_of_transfers[u] + 1

                            if new_d < distance[edge[1]] or (new_d == distance[edge[1]] and new_w < walking_time[edge[1]]) or (new_d == distance[edge[1]] and new_w == walking_time[edge[1]] and new_n < number_of_transfers[edge[1]]):
                                distance[edge[1]] = new_d
                                walking_time[edge[1]] = new_w
                                number_of_transfers[edge[1]] = new_n
                                previous_edge[edge[1]] = edge
                                
                            if edge[1] == self.source:
                                path_found = True
                                
            if path_found:
                distance_found = distance[self.source]
                
        if not previous_edge[self.source]:
            return None, None, None, None, None
        return self.source, previous_edge, distance[self.source], walking_time[self.source], number_of_transfers[self.source]


