import pandas as pd
import gzip
from collections import deque
import networkx as nx
import py4cytoscape as p4c
import numpy as np

# Sampled from YJMob100k dataset #2
file_path = "experiment_data/yjmob_sample.csv"

lines = []

# Read all lines
with open(file_path, 'r') as f:
    header = next(f).strip()  # read header
    for line in f:
        lines.append(line.strip())

# Create a dictionary from day intervals to their respective data
interval = 15
day_data_initial = {}
# for line in lines_to_keep:
for line in lines:
    user_id, day, time, x_coord, y_coord = line.split(',')
    day = int(day)
    interval_key = day // interval
    if interval_key not in day_data_initial:
        day_data_initial[interval_key] = []
    day_data_initial[interval_key].append((
        user_id,
        # Treat intervals as one continuous "day"
        (day % interval) * 48 + int(time),
        (float(x_coord), float(y_coord))
    ))
    
# Sort day_data_initial by key
day_data_initial = dict(sorted(day_data_initial.items()))

# Create dictionaries within each interval for their respective users
day_data_final = {}
# Iterate over (key, value) pairs in the interval dictionary
for interval_key, interval in day_data_initial.items():
    # Create a new dictionary for each interval
    user_dict = {}
    for record in interval:
        # Obtain user_id
        user_id = record[0]
        if user_id not in user_dict:
            user_dict[user_id] = []
        # Append the record to the user's list
        user_dict[user_id].append((
            record[1],
            record[2]
        ))
    # Assign the user dictionary to the finalized day data
    day_data_final[interval_key] = user_dict

# Now, day_data_final is structured as:
# { interval_key: { user_id: [ (time, (x coord, y coord)), ... ] } }

# Determine collisions within each interval (will correspond to edges in the graph)
#   A collision is defined as two users being in the same (x_coord, y_coord) within a 2 time unit window
# Collisions stored as {(user1, user2): count}
contact_networks = []
social_networks = []

print(day_data_final.keys())
print("Is every interval having the same exact data?", day_data_final[0] == day_data_final[1])

for interval_key, interval_data in day_data_final.items():
    collision_data = {}

    # Cartesian distance
    def distance(coords1, coords2):
        return ((coords1[0] - coords2[0]) ** 2 + (coords1[1] - coords2[1]) ** 2) ** 0.5

    # Iterate through each user and their time-coordinates
    for userID, user_data in interval_data.items():
        user_time_length = len(user_data)
        for t_index in range(user_time_length):
            # Since we care about collisions within a 2 time unit window,
            # check current and next coordinate
            cur_coord = user_data[t_index][1]
            next_coord = None
            if t_index == user_time_length - 1:
                next_coord = cur_coord  # No next coord, stay the same
            else:
                next_coord = user_data[t_index + 1][1]  # Get next coord

            # Compare to all other users
            for other_userID, other_user_data in interval_data.items():
                if other_userID == userID:
                    continue
                other_time_length = len(other_user_data)
                if other_time_length < user_time_length:
                    continue  # More comparisons happen when other user has fewer time points
                            # This avoid redundancy
                else:
                    other_coord = other_user_data[t_index][1]
                    # Check distance for current coords
                    if distance(cur_coord, other_coord) == 0 or distance(next_coord, other_coord) == 0:
                        # Record collision
                        edge = tuple(sorted((userID, other_userID)))
                        if edge not in collision_data:
                            collision_data[edge] = 1
                        else:
                            collision_data[edge] += 1

    # Create an undirected contact network from collision data
    contact_network = nx.Graph()
    for edge in collision_data.keys():
        contact_network.add_edge(edge[0], edge[1])

    # Compute the 75th percentile of the collision count distribution
    collision_counts = np.array(list(collision_data.values()))
    percentile_75 = np.percentile(collision_counts, 75)

    # Create social network from collision data: continue adding edges until entire graph is connected
    social_network = nx.Graph()
    for edge, count in collision_data.items():
        if count >= percentile_75:
            social_network.add_edge(edge[0], edge[1])

    # Add any missing nodes from contact network to social network
    for node in contact_network.nodes():
        if node not in social_network:
            social_network.add_node(node)

    # Rename nodes to be sequential integers starting from 0
    nodes = sorted(contact_network.nodes())
    mapping = {old_label: i for i, old_label in enumerate(nodes)}
    contact_network = nx.relabel_nodes(contact_network, mapping)
    social_network = nx.relabel_nodes(social_network, mapping)

    # Make the social network directed
    social_network = social_network.to_directed()

    # Import the networks into Cytoscape for visualization
    p4c.create_network_from_networkx(contact_network, title=f"YJMob Contact Network {interval_key}", collection="YJMob Networks")
    p4c.create_network_from_networkx(social_network, title=f"YJMob Social Network {interval_key}", collection="YJMob Networks")

    # Save networks to GML files to use in MFA
    nx.write_gml(contact_network, "experiment_data/mfa_contact.gml")
    nx.write_gml(social_network, "experiment_data/mfa_social.gml")

    contact_networks.append(contact_network)
    social_networks.append(social_network)