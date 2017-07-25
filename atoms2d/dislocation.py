import ase
import ase.geometry
import numpy as np

from . import atoms
from . import private


def core(type, a, elements):
    """Generate a dislocation core.

    Keyword arguments:
    type -- dislocation type (4|6, 5|7 or 6|8)
    a -- lattice constant
    elements -- array of elements with their symbols and z separation distance
    """
    positions = []
    symbols = []
    cell = [
        [a * 2, 0, 0],
        [0, 3 * a * np.sqrt(3) / 2, 0],
        [0, 0, 2 * a]
    ]

    def assign_atoms(polygon, next_atom=0):
        def get_next_atom(next_atom):
            if next_atom == 0:
                return 1
            return 0

        for index, p in enumerate(polygon.coords):
            if (not private.isEven(polygon.sides)
                    and np.ceil(polygon.sides / 2.0) == index):
                next_atom = get_next_atom(next_atom)

            positions.append([p[0], p[1], 0])
            symbols.append(elements[next_atom]['symbol'])

            if elements[next_atom].get('distance') > 0:
                positions[-1][-1] = -elements[next_atom]['distance'] / 2.0
                positions.append([p[0], p[1],
                                 elements[next_atom]['distance'] / 2.0])
                symbols.append(elements[next_atom]['symbol'])

            next_atom = get_next_atom(next_atom)

    if type == '4|6':
        square = private.Polygon(4, a)
        hexagon = private.Polygon(6, a)

        hexagon.translate([0, square.coords[2][1] - hexagon.coords[0][1]])
        assign_atoms(square)
        assign_atoms(hexagon)
    elif type == '5|7':
        pentagon = private.Polygon(5, a)
        heptagon = private.Polygon(7, a)

        heptagon.rotate(np.pi)
        heptagon.translate([0, pentagon.coords[3][1] - heptagon.coords[3][1]])
        assign_atoms(pentagon)
        assign_atoms(heptagon, 1)
    elif type == '6|8':
        hexagon = private.Polygon(6, a)
        octagon = private.Polygon(8, a)

        octagon.translate([0, hexagon.coords[2][1] - octagon.coords[1][1]])
        assign_atoms(hexagon)
        assign_atoms(octagon, 1)
        del positions[9]
        del positions[9]
        del symbols[9]
        del symbols[9]
    else:
        print(type + ' not supported')

    dislocation = atoms.Atoms(positions=positions, cell=cell)
    dislocation.set_chemical_symbols(symbols)
    dislocation.translate([a, cell[1][1] - dislocation.positions[0][1], a])
    ase.geometry.get_duplicate_atoms(dislocation, cutoff=0.5, delete=True)

    return dislocation


def line(type, primitive, rows, polarity=0):
    """Generate a line with a dislocation core.

    Keyword arguments:
    type -- dislocation type (4|6, 5|7 or 6|8)
    primitive -- the primitive cell of the material
    rows -- number of rows for the dislocation line
    polarity -- either TM first (0) or DC first (1) (default 0)
    """
    disloc_line = primitive.copy()
    a = disloc_line.lat_const[0]
    disloc_line.to_monolayer()
    symbols = set(disloc_line.get_chemical_symbols())
    elements = [{'symbol': '', 'distance': 0} for i in range(2)]

    # Store TM and DC symbols
    for s in symbols:
        if s == 'Mo' or s == 'W':
            elements[0]['symbol'] = s
        elif s == 'S' or s == 'Se':
            elements[1]['symbol'] = s

    # Get DC separation distance.
    high_dc = max(disloc_line.positions, key=lambda p: p[2])
    low_dc = min(disloc_line.positions, key=lambda p: p[2])
    elements[1]['distance'] = high_dc[2] - low_dc[2]

    disloc_line *= (2, 2 * (rows - 1), 1)
    disloc_line.to_orthorhombic()
    dz = max(disloc_line.positions, key=lambda p: p[1])[2]

    if polarity == 1:
        elements[0], elements[1] = elements[1], elements[0]
        disloc_line.reflect(0, along='y')

    dislocation = core(type, a, elements)

    # Move dislocation core to the correct z position.
    dz = dz - dislocation.cell[2][2] / 2
    dislocation.translate([0, 0, dz])

    # Move line to the bottom of the, will be created, core.
    top_atom = max(disloc_line.positions, key=lambda p: p[1])
    disloc_line.translate([-a / 2.0, -top_atom[1], 0])
    disloc_line.remove_atoms(-9999, 1e-4)

    disloc_line += dislocation

    if polarity == 1:
        disloc_line.reflect(0, along='y')

    # Correct cell size to include the dislocation core.
    disloc_line.set_cell([
        disloc_line.cell[0],
        [0, disloc_line.cell[1][1] + dislocation.cell[1][1], 0],
        disloc_line.cell[2]
    ])

    # Make the dislocation at the edge of the cell.
    if polarity == 1:
        disloc_line.translate([0, dislocation.cell[1][1], 0])
    else:
        disloc_line.translate([
            0, disloc_line.cell[1][1] - dislocation.cell[1][1], 0])

    return disloc_line
