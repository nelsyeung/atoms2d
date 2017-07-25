import ase.io
import numpy as np

from .atoms import Atoms


def read(file):
    atoms = ase.io.read(file)
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    lat_const = [0, 0, 0]

    if file.split('.')[-1] == 'cif':
        lat_const = [np.linalg.norm(np.array(cell[i])) for i in range(3)]

    return Atoms(cell=cell, positions=positions, numbers=numbers,
                 lat_const=lat_const, pbc=[1, 1, 0])
