import ase
import numpy as np


class Atoms(ase.Atoms):
    def __init__(self, symbols=None, positions=None, numbers=None, tags=None,
                 momenta=None, masses=None, magmoms=None, charges=None,
                 scaled_positions=None, cell=None, pbc=[1, 1, 0],
                 celldisp=None, constraint=None, calculator=None, info=None,
                 lat_const=[0, 0, 0]):
        super(Atoms, self).__init__(symbols=symbols, positions=positions,
                                    numbers=numbers, tags=tags,
                                    momenta=momenta, masses=masses,
                                    magmoms=magmoms, charges=charges,
                                    scaled_positions=scaled_positions,
                                    cell=cell, pbc=pbc, celldisp=celldisp,
                                    constraint=constraint,
                                    calculator=calculator, info=info)
        self.lat_const = lat_const

    def copy(self):
        atoms = super(Atoms, self).copy()
        atoms.lat_const = self.lat_const

        return atoms

    def rotate(self, angle, center='COU', rotate_cell=False):
        """Rotate the whole cell about the z axis by the specified angle in
        radians."""
        super(Atoms, self).rotate(angle, 'z', center=center,
                                  rotate_cell=rotate_cell)

    def to_monolayer(self):
        """Convert bulk to monolayer."""
        num_atoms = len(self.positions)
        cell_height = self.lat_const[2] / 2

        if cell_height == 0:
            cell_height = self.cell[2][2] / 2

        self.cell[2][2] = cell_height
        i = 0

        while i < num_atoms:
            if self.positions[i][2] > cell_height:
                self.pop(i)
                num_atoms -= 1
                i -= 1

            i += 1

    def to_orthorhombic(self):
        """Make unit cell orthorhombic."""
        for position in self.positions:
            if position[0] < 1e-4:
                multiplier = np.floor(abs(position[0]) / self.cell[0][0]) + 1
                position[0] += multiplier * self.cell[0][0]

        # Repeat twice to be more consistent
        for position in self.positions:
            if position[0] < 1e-4:
                multiplier = np.floor(abs(position[0]) / self.cell[0][0]) + 1
                position[0] += multiplier * self.cell[0][0]

        self.cell = [
            [self.cell[0][0], 0, 0],
            [0, self.cell[1][1], 0],
            self.cell[2]
        ]

    def reflect(self, x=0, along='x'):
        """Reflect the whole cell about a given line."""
        axis = 0

        if along == 'y':
            axis = 1

        for position in self.positions:
            position[axis] = 2 * x - position[axis]

        for vectors in self.cell:
            vectors[axis] = 2 * x - vectors[axis]

    def remove_atoms(self, gt, lt, along='x'):
        """Cut atoms along the specified direction that are greater than and
        less than the specified values."""
        num_atoms = len(self.positions)
        i = 0
        axis = 0

        if along == 'y':
            axis = 1
        elif along == 'z':
            axis = 2

        while i < num_atoms:
            if self.positions[i][axis] < lt and self.positions[i][axis] > gt:
                self.pop(i)
                num_atoms -= 1
                i -= 1

            i += 1

    def replace_overlaps(self, gb_loc):
        """Replace any overlapping atoms with their average position."""
        num_atoms = len(self.positions)
        lat_const = min(self.lat_const)
        i = 0

        while i < num_atoms:
            pos1 = self.positions[i]
            j = 0

            if pos1[0] < gb_loc - lat_const or pos1[0] > gb_loc + lat_const:
                i += 1
                continue

            while j < num_atoms:
                pos2 = self.positions[j]
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                dz = pos2[2] - pos1[2]

                if i != j and np.sqrt(dx*dx + dy*dy + dz*dz) < lat_const / 3:
                    pos2[0] = (pos1[0] + pos2[0]) / 2
                    pos2[1] = (pos1[1] + pos2[1]) / 2
                    pos2[2] = (pos1[2] + pos2[2]) / 2
                    self.pop(i)
                    num_atoms -= 1
                    i -= 1
                    j -= 1
                    break

                j += 1

            i += 1
