import numpy as np
import os

from . import atoms
from . import dislocation
from . import io
from . import private


def gb_distance(rows, lat_const):
    """Return the interspacing between the two adjacent dislocation cores."""
    return np.sqrt(3) * lat_const * (rows + 0.5)


def gb_angle(rows):
    """Return the misorientation angle in radians from the number of rows
    before repeat."""
    return np.degrees(np.arcsin(1.0 / (np.sqrt(3) * (rows + 0.5))))


def nr(cif, rows, columns, type, strain=0):
    """Generate a nanoribbon with a dislocation."""
    angle = gb_angle(rows) / 2
    tm = ''
    dc = ''

    primitive = io.read(os.path.join(os.getcwd(), cif + '.cif'))
    elements = set(primitive.get_chemical_symbols())

    # Store transition metal symbol and dichalcogenide symbol
    for e in elements:
        if tm == '' and (e == 'Mo' or e == 'W'):
            tm = e
        elif dc == '' and (e == 'S' or e == 'Se'):
            dc = e

    gb = primitive.copy()
    gb *= (columns, 2 * rows, 1)
    gb.to_monolayer()
    gb.to_orthorhombic()
    new_cell = [
        gb.cell[0],
        [0, gb_distance(rows, gb.lat_const[0]), 0],
        gb.cell[2],
    ]
    gb *= (3, 3, 1)
    gb.rotate(angle)
    gb.translate([-new_cell[0][0], -new_cell[1][1], 0])
    gb.set_cell(new_cell)
    gb.remove_atoms(-9999, 0)
    gb.remove_atoms(new_cell[0][0], 9999)
    gb.remove_atoms(-9999, 0, along='y')
    gb.remove_atoms(new_cell[1][1], 9999, along='y')

    nearest_gb = sorted(gb.positions, key=lambda p: p[0], reverse=True)

    for p in nearest_gb:
        i = private.where(gb.positions, p)

        if (gb.get_chemical_symbols()[i] == tm):
            nearest_gb = p
            break

    gb.translate([gb.cell[0][0] - nearest_gb[0] + strain / 4, 0, 0])

    disloc_line = dislocation.line(type, primitive, rows)
    top_tm = max(disloc_line.positions, key=lambda p: p[1])
    disloc_line.translate([gb.cell[0][0] - top_tm[0],
                           nearest_gb[1] - top_tm[1], 0])

    gb.remove_atoms(gb.cell[0][0] - gb.lat_const[0] + strain / 4 + 0.1, 9999)
    gb += disloc_line
    gb.wrap(pbc=(0, 1, 0))

    right = gb.copy()
    right.reflect(x=right.cell[0][0])
    gb += right

    return gb


def pbc_single(cif, rows, columns, type, strain=0, polarity=0):
    """Generate a grain boundary with a single dislocation and with periodic
    boundary condition."""
    primitive = io.read(os.path.join(os.getcwd(), cif + '.cif'))

    # Create a strip
    strip = primitive.copy()
    strip.to_monolayer()
    strip *= (1, 2 * rows + 1, 1)
    strip.to_orthorhombic()

    tm = ''
    dc = ''
    elements = set(primitive.get_chemical_symbols())

    # Store transition metal symbol and dichalcogenide symbol
    for e in elements:
        if tm == '' and (e == 'Mo' or e == 'W'):
            tm = e
        elif dc == '' and (e == 'S' or e == 'Se'):
            dc = e

    # Create a slowly rotating material from grain-boundary angle to 0.
    angle = gb_angle(rows) / 2
    dangle = angle / (columns - 1)
    lh = strip.copy()

    sign = -1 if polarity == 0 else 1

    for i in range(1, columns):
        new_strip = strip.copy()
        new_strip.rotate(-1 * i * dangle)
        new_strip.translate((sign * i * new_strip.lat_const[0], 0, 0))
        lh += new_strip

    # Update the new cell size and move structure into the cell
    cell_width = columns * primitive.lat_const[0]

    if polarity == 0:
        lh.translate([cell_width - lh.lat_const[0], 0, 0])
    else:
        lh.translate([-lh.lat_const[0] / 2 - cell_width, 0, 0])
        lh.reflect(0)

    lh.set_cell([
        [cell_width, 0, 0],
        lh.cell[1],
        lh.cell[2]
    ])

    # Get the top most atom from the structure
    if polarity != 0:
        nearest_gb = sorted(lh.positions, key=lambda p: p[1], reverse=True)

    # Get the left most atom from the structure
    nearest_gb = sorted(lh.positions, key=lambda p: p[0])

    check = dc if polarity == 0 else tm
    for p in nearest_gb:
        i = private.where(lh.positions, p)

        if (lh.get_chemical_symbols()[i] == check):
            nearest_gb = p
            break

    disloc_line = dislocation.line(type, primitive, rows, polarity=polarity)

    # Get the bottom/top most atom from the dislocation line
    disloc_sorted = sorted(disloc_line.positions, key=lambda p: p[1],
                           reverse=polarity != 0)

    bottom_disloc = disloc_sorted[0]
    top_disloc = disloc_sorted[-1]

    # Add the required extra atoms to the dislocation line
    extra_atom = tm if polarity == 0 else dc
    extra = atoms.Atoms(extra_atom,
                        [(primitive.lat_const[0], 0, top_disloc[2])],
                        cell=[1, 1, 1])

    if polarity == 1:
        extra += atoms.Atoms(extra_atom,
                             [(primitive.lat_const[0], 0,
                                 disloc_sorted[-2][2])],
                             cell=[1, 1, 1])
    extra.rotate(-angle)
    t = extra.copy()
    t.reflect(0)
    extra += t
    extra.translate([top_disloc[0], top_disloc[1], 0])
    disloc_line += extra

    # Move the dislocation line to one lattice constant away from the bottom
    # atom of the original structure
    disloc_width = primitive.lat_const[0] * np.cos(np.radians(angle / 2))
    dx = bottom_disloc[0] - nearest_gb[0]
    disloc_line.translate([-(disloc_width),
                           nearest_gb[1] - bottom_disloc[1], 0])

    # Make cell from the center of the dislocation line to the right edge of
    # the original structure
    lh.set_cell([
        [lh.cell[0][0] + dx - strain / 2, 0, 0], lh.cell[1], lh.cell[2]])
    lh.translate([dx - strain / 2, 0, 0])

    left = lh.copy()
    left.reflect(0)

    lh += disloc_line
    lh += left

    lh.translate([lh.cell[0][0], 0, 0])
    lh.set_cell([[2 * lh.cell[0][0], 0, 0], lh.cell[1], lh.cell[2]])
    lh.remove_atoms(lh.cell[0][0] - 0.01, 9999)

    return lh


def pbc(cif, rows, columns, type, strain=0):
    """Generate a grain boundary with two dislocations of opposite polarity and
    with periodic boundary condition."""
    primitive = io.read(os.path.join(os.getcwd(), cif + '.cif'))
    tm = ''
    dc = ''
    elements = set(primitive.get_chemical_symbols())

    # Store transition metal symbol and dichalcogenide symbol
    for e in elements:
        if tm == '' and (e == 'Mo' or e == 'W'):
            tm = e
        elif dc == '' and (e == 'S' or e == 'Se'):
            dc = e

    gb = nr(cif, rows, columns, type, strain)

    nearest_gb = sorted(gb.positions, key=lambda p: p[0], reverse=True)

    for p in nearest_gb:
        i = private.where(gb.positions, p)

        if (gb.get_chemical_symbols()[i] == dc):
            nearest_gb = p
            break

    disloc_line = dislocation.line(type, primitive.copy(), rows, polarity=1)
    bottom_dc = min(disloc_line.positions, key=lambda p: p[1])
    disloc_line.translate([nearest_gb[0] - bottom_dc[0] - strain / 4,
                           nearest_gb[1] - bottom_dc[1], 0])
    gb.remove_atoms(nearest_gb[0] - gb.lat_const[0] - strain / 4 + 0.1, 9999)
    gb += disloc_line
    gb.wrap(pbc=(0, 1, 0))

    right = gb.copy()
    right.reflect(x=nearest_gb[0] - strain / 4)
    gb += right

    gb.set_cell([
        [(nearest_gb[0] - strain / 4 - gb.cell[0][0]) * 2, 0, 0],
        gb.cell[1],
        gb.cell[2]
    ])

    gb.translate([-gb.cell[0][0] / 4, 0, 0])
    gb.remove_atoms(-9999, 0)
    gb.remove_atoms(gb.cell[0][0], 9999)

    return gb


def lh(A, B, rows, columns, type):
    """Generate a lateral heterostructure.

    Keyword arguments:
    A -- material with the larger lattice constant than B
    B -- material with the smaller lattice constant than A
    rows -- number of rows for both materials
    columns -- number of columns for both materials
    type -- dislocation type (4|6, 5|7 or 6|8)
    """
    # Calculate the strain required to create the structure
    A_test = pbc_single(A, rows, columns, type)
    B_test = io.read(B + '.cif')
    strain = A_test.cell[0][0] - 2 * (columns + 1) * B_test.lat_const[0]

    A_top = pbc_single(A, rows, columns, type, polarity=1, strain=strain)
    A_bottom = pbc_single(A, rows, columns, type, strain=strain)
    A_bottom.translate([0, A_top.cell[1][1], 0])
    structure = A_top.copy()
    structure += A_bottom
    structure.set_cell([
        structure.cell[0],
        structure.cell[1] * 2,
        structure.cell[2]
    ])

    dz = 0

    # Get A material TM z position
    for i in range(len(A_top.positions)):
        if (A_top.get_chemical_symbols()[i] == 'Mo' or
                A_top.get_chemical_symbols()[i] == 'W'):
            dz = A_top.positions[i][2]
            break

    B_top = io.read(B + '.cif')
    B_top *= (2 * (columns + 1), 2 * rows, 1)
    B_top.to_monolayer()
    B_top.to_orthorhombic()
    B_bottom = B_top.copy()

    # Get B material TM z position
    for i in range(len(B_top.positions)):
        if (B_top.get_chemical_symbols()[i] == 'Mo' or
                B_top.get_chemical_symbols()[i] == 'W'):
            dz -= B_top.positions[i][2]
            break

    B_top.translate([
        -B_bottom.lat_const[0] / 2,
        structure.cell[1][1],
        dz
    ])
    B_bottom.translate([
        -B_bottom.lat_const[0] / 2,
        -B_bottom.cell[1][1],
        dz
    ])
    structure += B_top
    structure += B_bottom

    structure.translate([0, B_bottom.cell[1][1], 0])
    structure.set_cell([
        structure.cell[0],
        structure.cell[1] + B_bottom.cell[1] * 2,
        structure.cell[2]
    ])

    return structure
