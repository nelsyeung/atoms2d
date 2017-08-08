from __future__ import print_function
import networkx as nx

from . import private


def relax(atoms, lat_const):
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
        'top': sorted_pos[1][0][0] - (lat_const / 2 + correction),
        'right': sorted_pos[0][-1][0] - (lat_const / 2 + correction),
        'bottom': sorted_pos[1][0][1] + (lat_const / 2 + correction),
        'left': sorted_pos[0][0][0] + (lat_const / 2 + correction)
    }

    for i in range(len(atoms.positions)):
        graph.add_node(i, fixed=False, position=atoms.positions[i][:],
                       symbol=symbols[i])

        if (atoms.positions[i][1] > boundaries['top'] or
                atoms.positions[i][0] > boundaries['right'] or
                atoms.positions[i][1] < boundaries['bottom'] or
                atoms.positions[i][0] < boundaries['left']):
            graph.node[i]['fixed'] = True

    # Form edges between all nearby atoms
    loop = 0
    total_loops = float(num_atoms * num_atoms)
    print('Creating edges...')

    for i in atoms_range:
        for j in atoms_range:
            progress = '{0:.0f}%'.format(100.0 * float(loop) / total_loops)
            print(progress, end='\r')
            loop += 1

            if (i == j or
                    not private.is_same_plane(atoms.positions[i],
                                              atoms.positions[j]) or
                    not private.is_same_type(symbols[i], symbols[j])):
                continue

            distance = atoms.get_distance(i, j)

            if (distance < 1.5 * lat_const):
                graph.add_edge(i, j)

    progress = '{0:.0f}%'.format(100.0 * float(loop) / total_loops)
    print(progress)

    # Move atoms to centroid of each connected nodes
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
            node = graph.node[j]
            centroid[0] += node['position'][0]
            centroid[1] += node['position'][1]

        centroid[0] /= num_edges
        centroid[1] /= num_edges

        # Move atom to the centroid
        atoms.positions[i][0] += centroid[0] - graph.node[i]['position'][0]
        atoms.positions[i][1] += centroid[1] - graph.node[i]['position'][1]

    # Update nodes position to their new position
    for i in atoms_range:
        graph.node[i]['position'] = atoms.positions[i][:-1]

    return atoms
