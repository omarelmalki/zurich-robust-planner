import networkx as nx
import datetime
import copy
import pandas as pd
import scipy.stats as stats
from database import Database
from utils import get_env_vars
import matplotlib.pyplot as plt
import ast
from datetime import timedelta
import numpy as np
import ipywidgets as widgets
from datetime import date
import random
import plotly.graph_objects as go
import numpy as np


# #### Interface

def build_input(stops_df):
    '''
        Build all widgets and combine it to an input interface.Return interface and wigets, so that later on the value can be read.
        
        Output: interface, input_success, input_arrival, input_departure, input_time, number_input
    '''
    # Build widgets
    today = date.today()
    stop_names = stops_df.stop_name.unique()
    style = {'description_width': 'initial'}

    input_departure = widgets.Combobox(
        placeholder='From',
        options=list(stop_names),
        description='Departure stop:',style=style
    )

    input_arrival = widgets.Combobox(
        placeholder='To',
        options=list(stop_names),
        description='Arrival stop:',style=style
    )

    input_date = widgets.DatePicker(
        description='Travel date',style=style, value = today    
    )

    input_time = widgets.TimePicker(
        description='Arrival time', value = datetime.datetime.now().time())

    input_success = widgets.IntSlider(
        value=95,
        min=80,
        max=99,
        step=1,
        description='Probability of success:',
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style = style
    )
    
    number_input = widgets.FloatText(description='Number of routes shown:', value=3, style=style, layout=widgets.Layout(width='25%'))

    # Combine widgets
    input_travel =  widgets.HBox([input_departure, input_arrival, input_time])
    input_start = widgets.HBox([input_success, number_input])
    interface = widgets.VBox([input_travel, input_start])
    return interface, input_success, input_arrival, input_departure, input_time, number_input

def get_choosetrip_w(route):
    '''
        Build widgets to specify route we want to get more details about. 
    '''
    # Widget to choose a trip for detailed info
    style = {'description_width': 'initial'}
    choose_trip = widgets.Dropdown(
        options= list(np.arange(0,len(route))+1),
        value=1,
        description='Get details about route:',
        disabled=False,style=style, layout=widgets.Layout(width='25%')
    )
    return choose_trip


# ### Visualization

# ##### Figure 1

def timetable_overview(route, travel_success, n_trips):
    """ 
    Generates an overview figure of all computed routes. 
        Explanation of figure:
            Walking: dashed line. Only visualized if it is between different stops.
            Points: Public transport stop
            Line length: Percentage of travel duration compared to the travel duration of the whole travel time of the corresponding route.
            Dark blue and bold line: Routes that meet p(success)
            Light blue line: Routes that don't meet p(success)
            
    Input: 
            route: Route from the route planning algorithm
            travel_success: Set from the user. p(success) that has to be met 
            n_trips: number of trips the user wants to see
    Output: 
            figure: The figure explained above
            stops_tot: Stops of shortest route for each route (Each route is a row)
            transport_type: Edge type (train, walk, bus ect.) of each edge in shortest route for each route 
            arrival_tot: departure_tot: Arrival resp. deprarture time for each stop (for each trip)
            x_tot: Percentage of how much each trip segment is compared to whole trip. X axis for figure.
            keep_index_tot: True if station is an arrival/ departure stop and not an intermediate stop.
            route_name_tot: The name of the route if available e.g. S14
            headsign_tot: Headsign of the route if available (Corresponds to the travel direction)
            stops_id_tot: Stop id  of shortest route for each route (Each route is a row)
            trip_id_tot: Trip id of each route"""
    

    # Create figure to show color of label-----------------------------------
    fig_label, ax_label = plt.subplots(figsize=(3, 0.5))

    # Plot the first horizontal line
    ax_label.hlines(y=0.5, xmin=0, xmax=0.5, color="#1B62A9", linewidth=2)
    ax_label.text(0.6, 0.5, 'Fulfills p(success)', ha='left', va='center')

    # Plot the second horizontal line
    ax_label.hlines(y=0, xmin=0, xmax=0.5, color="#A8C5E2", linewidth=2)
    ax_label.text(0.6, 0, 'Does not meet p(success)', ha='left', va='center')

    plt.axis('off')
    plt.axis('equal')
    # -----------------------------------
    
    
    route_viz = route.copy() # To avoid changing original data
    # Data storage position from shortest path algorithm result
    data_storage = 0
    p_succes_storage = 1
    detail_info = -1

    p_success_tot = []
    stops_tot = []
    stops_id_tot = []
    transport_type = []
    transport_subtype = []
    arrival_tot = []
    departure_tot = []
    x_tot = []
    keep_index_tot = []
    type_edge_tot = []
    route_name_tot = []
    walking_t_tot = []
    headsign_tot = []
    trip_id_tot = []


    figure, ax = plt.subplots(figsize=(15,5))

    # Visualize best trips 
    for trip_n in range(n_trips):
        stops_trip = []
        stop_id_trip = []
        type_edge = []
        route_name = []
        arrival_target = [] # arrival time of target node
        departure = []
        keep_index = [] # Extracts stops of departure/ transit/ arrival into transit_index for visualization
        travel_time = []
        walking_time = []
        headsign = []
        trip_id = []
        p_success =  round(route_viz[trip_n][p_succes_storage],2)


        # Go through each stop of a trip and extract info----------------------------------------------------------------------------------------
        for stop_n in range(len(route_viz[trip_n][data_storage])):

            # Ignore stop if it is a transit within same station: a walking edge with same departure and arrival stop
            # our input interface asks for station name. We assign station_id of parent when info is available. This catches cases where this is not the case 
            if route_viz[trip_n][data_storage][stop_n][detail_info]['transport_type'] == 'walk' and  route_viz[trip_n][data_storage][stop_n][detail_info]['stop_name'] == route_viz[trip_n][data_storage][stop_n][detail_info]['next_stop_name'] :
                continue


            # Extract position in list of departure/ transit/ arrival-stop into keep_index for visualization
            if stop_n == 0: # First stop is always true
                keep_index.append(True)   
            elif (route_viz[trip_n][data_storage][stop_n][detail_info]['trip_id'] != route_viz[trip_n][data_storage][stop_n-1][detail_info]['trip_id']): # Trip id changes -> transit stop
                keep_index.append(True)
            else:
                keep_index.append(False)


            #  Calculate arrival time of walkable edges extract walking time
            if route_viz[trip_n][data_storage][stop_n][detail_info]['is_walkable']:
                walking_time.append(route_viz[trip_n][data_storage][stop_n][detail_info]['duration_s'])
                if stop_n < len(route_viz[trip_n][data_storage])-1: # Not the last stop -> arrival is departure of next transport typs
                    arrival_target.append(route_viz[trip_n][data_storage][stop_n+1][detail_info]['departure_time'])
                else: # Calculate it based on walking time
                    arrival_walk = datetime.datetime.strptime(route_viz[trip_n][data_storage][stop_n][detail_info]['departure_time'] , '%H:%M:%S') + timedelta(seconds=route_viz[trip_n][data_storage][stop_n][detail_info]['duration_s'])
                    arrival_target.append(arrival_walk.strftime('%H:%M:%S'))
            else:
                arrival_target.append(route_viz[trip_n][data_storage][stop_n][detail_info]['next_arrival_time'])
            
            '''
            # Only add stop_id if it is not walkable -> for visualization 3
            if route_viz[trip_n][data_storage][stop_n][detail_info]['is_walkable'] != True:
                stop_id_trip.append(route_viz[trip_n][data_storage][stop_n][1])'''


            # Extract rest of the information
            stops_trip.append(route_viz[trip_n][data_storage][stop_n][detail_info]['stop_name'])
            stop_id_trip.append(route_viz[trip_n][data_storage][stop_n][1])
            type_edge.append(route_viz[trip_n][data_storage][stop_n][detail_info]['transport_type'])
            route_name.append(route_viz[trip_n][data_storage][stop_n][detail_info]['route_name'])
            travel_time.append(route_viz[trip_n][data_storage][stop_n][detail_info]['duration_s']/60)
            departure.append(route_viz[trip_n][data_storage][stop_n][detail_info]['departure_time'])
            headsign.append(route_viz[trip_n][data_storage][stop_n][detail_info]['trip_headsign'])
            trip_id.append(route_viz[trip_n][data_storage][stop_n][detail_info]['trip_id'])



        # Add last target node information
        stops_trip.append(route_viz[trip_n][data_storage][stop_n][detail_info]['next_stop_name'])
        stop_id_trip.append(route_viz[trip_n][data_storage][stop_n][0])
        keep_index.append(True) # Last arrival stop

        # Add info of each trip----------------------------------------------------------------------------------------
        stops_tot.append(stops_trip)
        stops_id_tot.append(stop_id_trip)
        transport_type.append(type_edge)
        arrival_tot.append(arrival_target)
        departure_tot.append(departure)
        keep_index_tot.append(keep_index)
        type_edge_tot.append(type_edge)
        route_name_tot.append(route_name)
        walking_t_tot.append(walking_time)
        headsign_tot.append(headsign)
        trip_id_tot.append(trip_id)
        p_success_tot.append(p_success)

        # Calculate info for visualization of each trip ------------------------------------------------------------------
        # Extract departure/ transit/ arrival stops (Where point will be visualized) of  a trip
        stops_extracted =  [stops_trip for stops_trip, keep in zip(stops_trip, keep_index) if keep]
        arrival_extracted = [type_edge for type_edge, keep in zip(arrival_target, keep_index[1:]) if keep]
        type_extracted = [type_edge for type_edge, keep in zip(type_edge, keep_index) if keep]
        departure_extracted = [departure for departure, keep in zip(departure, keep_index[:]) if keep]

        start = datetime.datetime.strptime(departure_extracted[0], '%H:%M:%S')
        end = datetime.datetime.strptime(arrival_extracted[detail_info], '%H:%M:%S')
        duration = end-start
        total_seconds = duration.total_seconds()
        travel_minutes = int((total_seconds /60))

        # Calculate percentage of travel time for each trip segment
        x_i = [0] # First stop starts at 0 percent
        for i in range (len(arrival_extracted)):
                difference_t = datetime.datetime.strptime(arrival_extracted[i], '%H:%M:%S') - start 
                x_i.append(difference_t.total_seconds()/total_seconds) 

        x_tot.append(x_i)

        # Visualize trip------------------------------------------------------------------------------------------------------------------------------------
        # Add point for each arrival/ departure station 
        if p_success < travel_success/100: # Does not meet users p(success) input
            color = "#A8C5E2"
            width = 1
        else:
            color = "#1B62A9"
            width = 3

        for i in range(len(arrival_extracted)):
            y = [-0.1*trip_n] * 2
            x = x_i[i:i+2]
            # Walking edge has dotted line
            if type_extracted[i]=="walk":
                ax.plot(x, y, '--o', color = color, linewidth=width)
            else:
                ax.plot(x, y, '-o', color = color, linewidth=width)

        # Add labels on both ends of the line
        start_time_l = start.strftime('%H:%M:%S')
        start_label = f"Route {trip_n+1}     \n"+ start_time_l + "    " # Add space so that label is not too close to figure
        total_walking = round(sum(walking_t_tot[trip_n])/60)
        end_label = f"    Travel time: {travel_minutes} minutes, Total walking time: {total_walking} min"+ f", P(success): {p_success}" + "\n    "   + arrival_extracted[detail_info] 

        ax.text(x_i[0], y[0], start_label, ha='right', va='bottom')  # Label at the start
        ax.text(x_i[-1], y[-1], end_label, ha='left', va='bottom')      # Label at the end
        plt.axis('off')
        plt.axis('equal')
    return figure, stops_tot, transport_type , arrival_tot, departure_tot, x_tot, keep_index_tot, type_edge_tot, route_name_tot, headsign_tot, stops_id_tot, trip_id_tot


# ##### FIgure 2

# +
def generate_random_color(colors):
    '''
    Generate random color but make sure that it is not too dark or too light. Check that the generated color is not similar to any of the colors
    in the input colors-list.

    Input: list of colors already in usage
    Output: newly generated random color.
    '''
    while True:
        # Generate random values for red, green, and blue channels
        r = random.randint(30, 245)
        g = random.randint(30, 245)
        b = random.randint(30, 245)

        # Convert the values to hexadecimal strings and concatenate them
        color_hex = '#{:02x}{:02x}{:02x}'.format(r, g, b)

        # Check if the generated color is too similar to any of the existing colors
        too_similar = False
        for existing_color in colors:
            if color_distance(color_hex, existing_color) < 0.5:  # Adjust the threshold as needed
                too_similar = True
                break

        if not too_similar and color_hex != "#000000":
            return color_hex

# Function to calculate the color difference (Euclidean distance) between two colors
def color_distance(color1, color2):
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5



# -

def detailed_viz1(choose_trip, stops_tot, route_name_tot, headsign_tot, trip_id_tot, keep_index_tot, type_edge_tot, departure_tot, arrival_tot):
    '''
        Generates a detailed figure for a chosen route. It shows all the stop-names (including intermediate stops). The departure/ arrival time
        of each route segment, with route name and direction if this information is available.
        
        Input: 
            choose_trip: The widget from the overfiew figure where the user chooses the route he want to know more about. 
            stops_tot, route_name_tot, headsign_tot, trip_id_tot, keep_index_tot, type_edge_tot, departure_tot, arrival_tot: Output from the overview figure.
        
        Return:
            y_label: The stop names displayed 
            color_segment: Contains color of the route segment
            departure_seg, arrival_seg, name_seg, line_style: Attributes of the segment
            figure: Figure generated 
    '''
    # Extract input value
    detail_trip =choose_trip.value -1 

    # Split labels according to route segment and generate color for each segment
    # Get departure/ arrival and transport type of each segment
    y_label = []
    color_segment = []
    color = []
    departure_seg = []
    arrival_seg = []
    perron_seg = []
    headsign = []
    name_seg = []
    line_style = [] # 0 is dotted, 1 is line
    style_transport = 1
    style_walk = 0
    start_true = 1
    start_i = 0 # Index to extract a trip segment
    end_i = 0
    y_label = []
    for stop_n in range(0, len(keep_index_tot[detail_trip])):
        if keep_index_tot[detail_trip][stop_n] == True: 
            if start_true == 1:
                start_true = start_true*-1
            else:
                end_i = stop_n
                if type_edge_tot[detail_trip][start_i] == 'walk': 
                    line_style.append(style_walk)
                else:
                    line_style.append(style_transport)
                y_label.append(stops_tot[detail_trip][start_i:end_i+1]) 
                color_gen = generate_random_color(color)
                color.append(color_gen) # Ensures that next color is different
                color_segment.append(color_gen)
                departure_seg.append(departure_tot[detail_trip][start_i]) 
                arrival_seg.append(arrival_tot[detail_trip][end_i-1])
                headsign.append(headsign_tot[detail_trip][start_i]) 
                name_seg.append((route_name_tot[detail_trip][start_i]))
                start_i = end_i 

    # Create y-axis data
    figure, ax = plt.subplots(figsize=(5,10))
    y_start = 0
    for iter_i, y_label_i in enumerate(y_label):    
        x_data = [0] * len(y_label_i)
        y_data = -1*np.arange(0,len(y_label_i),1) + y_start

        y_start = y_data[-1]-1
        # Create labels for each point
        labels = y_label_i

        # Plot the line with connected points
        if line_style[iter_i] == style_walk:
            ax.plot(x_data, y_data,  linestyle='--', c='black')
        else:
            ax.plot(x_data, y_data, marker='o', linestyle='-', c=color_segment[iter_i])

        # Add labels to each point on right side
        for x, y, label in zip(x_data, y_data, labels):
            ax.text(x, y, "   "+ label, ha='left', va='center')

        # Add left side of graph: transport type, departure time, arrival time
        if name_seg[iter_i]  == None and headsign[iter_i] == None:
            ax.text(x_data[0], y_data[0], departure_seg[iter_i] + "   " , ha='right', va='center')
        elif name_seg[iter_i]  == None:
            ax.text(x_data[0], y_data[0], f" in direction {headsign[iter_i]} \n" +departure_seg[iter_i] + "   " , ha='right', va='center')
        elif headsign[iter_i] == None:
            ax.text(x_data[0], y_data[0], name_seg[iter_i] + f"\n" +departure_seg[iter_i] + "   " , ha='right', va='center')
        else:   
            ax.text(x_data[0], y_data[0], name_seg[iter_i] + f" in direction {headsign[iter_i]} \n" +departure_seg[iter_i] + "   " , ha='right', va='center')

        ax.text(x_data[0], y_data[-1], arrival_seg[iter_i] + "   ", ha='right', va='bottom')

    plt.axis('off')
    plt.axis('equal')

    #plt.show()
    return y_label, color_segment, departure_seg, arrival_seg, name_seg, line_style, figure 


# ##### Figure 3

def build_coordinate_graph(railway_route):
    '''
        Builds a network in order to visualize a potential train railway route.
        Input: railway_route df with information about railway segments
        Output: 
            graph_coordinate: A network containing information of the SBB railway segments
    '''
    railway_route['BP Anfang.1'] =railway_route['BP Anfang.1'].astype(str).apply(lambda x: x.strip().lower())
    railway_route['BP Ende.1'] = railway_route['BP Ende.1'].astype(str).apply(lambda x: x.strip().lower())

    # Build graph for railway route (SBB data). Edges are railway route segments. Find shortest path between stations to get the railway route
    # Create an empty graph
    graph_coordinate = nx.Graph()

    # Iterate over the rows of the DataFrame
    for _, row in railway_route.iterrows():
        start_node = row['BP Anfang.1']
        end_node = row['BP Ende.1']
        attribute = row['Geo Shape']

        # Add an edge between the start and end nodes with the attribute as an edge attribute
        graph_coordinate.add_edge(start_node, end_node, attribute=attribute)
    return graph_coordinate


# +
def detailed_viz2(choose_trip, y_label, color_segment, departure_seg, arrival_seg, travel_name, stops_id_tot, stops_df, railway_route, stops_tot, graph_coordinate, line_style):
    '''
    Builds a map with colors corresponding to the figure built from detailed_viz1. 
    Output:
        fig: Returns the map
    Input: 
        choose_trip: The widget from the overfiew figure where the user chooses the route he want to know more about. 
        y_label: The stop names displayed in the previous figure
        color_segment, departure_seg, arrival_seg, travel_name, stops_id_tot: Attributes about each route segment
        stops_df: A df containing all information about stops. Used to get the stops coordinates and to check if railway coordinate has enough information for the corresponding route segment.
        railway_route: df with information about railway segments
        stops_tot: All visualized stops in overview figure
        graph_coordinate: Network of railway coordinates segments 
        line_style: Attribute of linestyle for each route segment 
    
    '''
    travel_name = ['No name available' if item is None else item for item in travel_name] # Replace None for labeling

    # Get coordinates of all stops to visualize hover
    detail_trip = choose_trip.value -1
    all_stop_viz = stops_df[stops_df.stop_id.isin(stops_id_tot[detail_trip])]


    # Check for which station information is missing
    not_in_railway = stops_df[~(stops_df.stop_name.isin(railway_route["BP Anfang.1"])|stops_df.stop_name.isin(railway_route["BP Ende.1"]))]

    coordinates = []
    coordinates_walk = []
    lon_min = []
    lon_max = []
    lat_min = []
    lat_max = []
    style_line = []
    train_style = 2 # Show dots
    tram_bus_style = 1 # Connect by line
    walk_style = 0 # dotted line
    coordinate_missing_railway = [] # special case where railway data is missing stop
    color_missing_railway = []
    color_transport = []
    name_label = []

    for row_i, route_segment in enumerate(y_label):
        # Check that no station is missing in railway SBB data. Then save coordinates of railway. No data was found for bus/tram lines -> just connect those stations
        if any(value in list(not_in_railway.stop_name) for value in y_label[row_i]) == False: 
            color_transport.append(color_segment[row_i])
            style_line.append(train_style) # Simply show points instead of connected points
            # Find path of railway route
            shortest_path = []
            for i in range(len(y_label[row_i])-1):
                start = y_label[row_i][i]
                end = y_label[row_i][i+1]
                segment_path = nx.shortest_path(graph_coordinate, start, end)
                # Check if data seems wrong
                # Some edge: data is missing, e.g. between 'dietikon schöneggstrasse', 'dietikon' -> algo would go backward of the train direction to find other path
                # Solution: If shortest path goes back 2 stops, we draw a straight line instead
                common_values = set(segment_path).intersection(y_label[row_i])
                if len(common_values) >= 4: 
                    missing_coord = stops_df[stops_df.stop_name.isin([start, end])].drop_duplicates("stop_name")[['stop_lon', 'stop_lat']].values.tolist()
                    coordinate_missing_railway.append(missing_coord)
                    color_missing_railway.append(color_segment[row_i])
                else:
                    shortest_path.extend(segment_path)


            # Get coordinates of shortest path
            coordinate_segment = []
            for i in range(len(shortest_path) - 1):
                u = shortest_path[i]
                v = shortest_path[i+1]
                if u!= v:
                    attrs = str(graph_coordinate[u][v])

                # Only keep cooridinates, transform to list
                attribute = attrs.split("[", 1)[-1]
                index_end = attribute.rfind("]")  # Find the index of the last occurrence of "]"
                attribute = ast.literal_eval(attribute[:index_end])  # Slice the string up to the last "]"
                coordinate_segment.extend(list(attribute))
            coordinates.append(coordinate_segment)
            name_label.append(travel_name[row_i])
        # No data about railway route available -> Connect points with line
        elif line_style[row_i] == walk_style:
            viz_station = stops_df[stops_df.stop_name.isin(route_segment)]
            viz_station = viz_station.drop_duplicates("stop_name")
            viz_station = viz_station.sort_values(by='stop_name', key=lambda x: x.map({v: i for i, v in enumerate(route_segment)}))
            coordinates_walk.append(viz_station[['stop_lon', 'stop_lat']].values.tolist())
        else: # tram, bus
            style_line.append(tram_bus_style) 
            viz_station = stops_df[stops_df.stop_name.isin(route_segment)]
            viz_station = viz_station.drop_duplicates("stop_name")
            viz_station = viz_station.sort_values(by='stop_name', key=lambda x: x.map({v: i for i, v in enumerate(route_segment)}))
            coordinates.append(viz_station[['stop_lon', 'stop_lat']].values.tolist())
            name_label.append(travel_name[row_i])
            color_transport.append(color_segment[row_i])



    # Define the coordinates for each scatter plot
    frame = stops_df[stops_df.stop_name.isin(stops_tot[detail_trip])]
    lon_min, lon_max = frame.stop_lon.min(), frame.stop_lon.max()
    lat_min, lat_max = frame.stop_lat.min(), frame.stop_lat.max()


    # Create a subfigure with multiple scatter_mapbox plots
    fig = go.Figure()

    # Add public transport line
    for i, coord in enumerate(coordinates):
        # Create a DataFrame from the coordinates
        coord_df = pd.DataFrame(coord, columns=['Longitude', 'Latitude'])
        if style_line[i] == train_style:
            style_attribute = 'markers'
        elif style_line[i] == tram_bus_style:
            style_attribute = 'lines'
        # Add the scatter map to the subfigure
        fig.add_trace(go.Scattermapbox(
        mode=style_attribute,
        lon=coord_df['Longitude'],
        lat=coord_df['Latitude'],
        line=dict(width=4, color=color_transport[i]),  
        hoverinfo='none',
        name= f'{name_label[i]}' 
    ))

    # Add walking 
    for i, coord in enumerate(coordinates_walk):
        # Create a DataFrame from the coordinates
        coord_df = pd.DataFrame(coord, columns=['Longitude', 'Latitude'])
        # Add the scatter map to the subfigure
        fig.add_trace(go.Scattermapbox(
        mode='lines',
        lon=coord_df['Longitude'],
        lat=coord_df['Latitude'],
        line=dict(width=4, color='black'),  # dash='dot'
        hoverinfo='none',
        showlegend=False
        ))

    # Add special cases where railway data is missing    
    if len(coordinate_missing_railway) >0:    
        for i, coord in enumerate(coordinate_missing_railway):
            # Create a DataFrame from the coordinates
            coord_df = pd.DataFrame(coord, columns=['Longitude', 'Latitude'])
            color_i = color_missing_railway[i]
            style_attribute = 'lines'

            # Add the scatter map to the subfigure
            fig.add_trace(go.Scattermapbox(
                mode='lines',
                lon=coord_df['Longitude'],
                lat=coord_df['Latitude'],
                line=dict(width=4, color=color_i),   
                hoverinfo='none',
                showlegend=False
            ))


    # Add black station position with hover-name 
    fig.add_trace(go.Scattermapbox(
        mode='markers',
        lon=all_stop_viz['stop_lon'],
        lat=all_stop_viz['stop_lat'],
        name = "All stops",
        hoverinfo = 'text', 
        text= all_stop_viz.stop_name,
        marker=dict(
        size=10,  
        color='black')
        ))

    fig.update_layout(
        mapbox=dict(
            style='carto-positron',  # Change the map style to black and white (carto-positron)
            center=dict(lon=(lon_min + lon_max) / 2, lat=(lat_min + lat_max) / 2),  # Set the center of the map
            zoom=11),
        margin={'r':0,'t':0,'l':0,'b':0}
        )


    # Display the map
    fig.show()
    return fig





