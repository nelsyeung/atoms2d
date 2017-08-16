from __future__ import print_function
import networkx as nx

from . import private


def relax(atoms, lat_const, steps=1, fixed=[]):
    """Relax a structure by minimizing the distance between each atom where the
    edge atoms are all fixed."""
    # Add all atoms as nodes to a graph
    graph = nx.Graph()
    num_atoms = len(atoms)
    atoms_range = range(num_atoms)
    symbols = atoms.get_chemical_symbols()
    correction = 0.1
    sorted_pos = [sorted(atoms.positions, key=lambda p: p[0]),
                  sorted(atoms.positions, key=lambda p: p[1])]
    boundaries = {
        'top': sorted_pos[1][-1][1] - (lat_const / 2 + correction),
        'right': sorted_pos[0][-1][0] - (lat_const / 2 + correction),
        'bottom': sorted_pos[1][0][1] + (lat_const / 2 + correction),
        'left': sorted_pos[0][0][0] + (lat_const / 2 + correction)
    }

    for i in atoms_range:
        position = atoms.positions[i][:]
        graph.add_node(i, fixed=False, position=position)

        if (position[1] > boundaries['top'] or
                position[0] > boundaries['right'] or
                position[1] < boundaries['bottom'] or
                position[0] < boundaries['left']):
            graph.node[i]['fixed'] = True
        else:
            for f in fixed:
                if all([abs(position[x] - f[x]) <= 1e-4 for x in range(2)]):
                    graph.node[i]['fixed'] = True

    # Form edges between all nearby atoms
    radius = 1.5 * lat_const
    for i in atoms_range:
        for j in atoms_range:
            if (i == j or
                    not private.is_same_plane(atoms.positions[i],
                                              atoms.positions[j]) or
                    not private.is_same_type(symbols[i], symbols[j])):
                continue

            distance = atoms.get_distance(i, j)

            if (distance < radius):
                graph.add_edge(i, j)

    # Move atoms to centroid of each connected nodes
    for _ in range(steps):
        for i in atoms_range:
            edges = graph[i]
            num_edges = len(edges)

            if (graph.node[i]['fixed'] or
                    num_edges < 6 or
                    not private.is_even(num_edges)):
                continue

            # Calculate centroid
            centroid = [0, 0]

            for j in edges:
                centroid[0] += graph.node[j]['position'][0]
                centroid[1] += graph.node[j]['position'][1]

            centroid[0] /= num_edges
            centroid[1] /= num_edges

            # Move atom to the centroid
            atoms.positions[i][0] += centroid[0] - graph.node[i]['position'][0]
            atoms.positions[i][1] += centroid[1] - graph.node[i]['position'][1]

        # Update nodes position to their new position
        for i in atoms_range:
            graph.node[i]['position'] = atoms.positions[i][:]

    return atoms
