import community

from Tools.get_graph import load_graph


def load_real_communities(path):
    """
    Load ground-truth communities from a real-world dataset label file.

    The file is expected to contain one node-community assignment per line:
        node_id community_id

    Args:
        path (str): Path to the ground-truth label file.

    Returns:
        list[list[int]]: A list of communities, where each community is a list of node IDs.
    """
    # Initialize a dictionary to store nodes grouped by community ID.
    community_dict = {}

    # Read the label file line by line.
    with open(path, 'r') as file:
        for line in file:
            # Split each line into node ID and community ID.
            node, community = line.strip().split()
            node = int(node)
            community = int(community)

            # Append the node to the corresponding community list.
            if community not in community_dict:
                community_dict[community] = []
            community_dict[community].append(node)

    # Convert the dictionary values to a nested list format.
    return list(community_dict.values())


def load_lfr_communities(path):
    """
    Construct reference communities for an LFR-style graph based on leaf-node attachment.

    For each non-leaf node (degree > 1), a community is formed containing:
    - the node itself
    - all of its leaf neighbors (degree == 1)

    Args:
        path (str): Path to the graph file.

    Returns:
        list[list[int]]: A list of communities constructed from the graph structure.
    """
    G = load_graph(path)
    communities = []

    # Traverse all nodes in the graph.
    for node in G:
        # Skip leaf nodes; communities are centered on non-leaf nodes.
        if len(G[node]) == 1:
            continue

        # Start a new community with the current non-leaf node.
        c = [node]

        # Add all leaf neighbors of the current node to the same community.
        for neighbor in G[node]:
            if len(G[neighbor]) == 1:
                c.append(neighbor)

        communities.append(c)

    return communities