import numpy as np
import pdb


class Atoms:
    ''' Class that track the featurization of atoms. There are so far two
    features possible for each atom:
    - the type of the atom (C, H...) that is given by a number between 1 and
    'self.num_of_types' (for example, C = 1 and H = 2).
    - the number of bonds that an atom has with the different types of atoms. 
    Keeping the example of the previous point, an atom bonded to 3 C and 1 H 
    will have features [3, 1]. The order of the features corresponds to the 
    order of the atom types (bonds with C comes first as C = 1, bonds with H
    comes 2nd as H = 2).
    '''

    def __init__(self, atom_features, num_of_types, atoms_type):
        self.atom_features = atom_features
        self.num_of_types = num_of_types
        self.num_atom_descriptors = self.atom_features[0] \
                                    + self.num_of_types*self.atom_features[1]
        self.atoms_type = atoms_type

    def get_atom_features_for_reaction(self, reax, real_bonds):
        # Compute the atom descriptions of 2 atoms involved in a bond change.
        # Real_bonds need to be symmetric.
        atom_description_1 = self.get_atom_features(reax[0], real_bonds)
        atom_description_2 = self.get_atom_features(reax[1], real_bonds)
        m = np.argwhere(atom_description_1[0] - atom_description_2[0] != 0)
        if m.shape[0] == 0 \
            or atom_description_1[0][min(m)] < atom_description_2[0][min(m)]:
            atom_description = np.concatenate((atom_description_1[0], 
                                               atom_description_2[0]))
        else:
            atom_description = np.concatenate((atom_description_2[0],
                                               atom_description_1[0]))
        return atom_description

    def get_atom_features(self, atom, real_bonds):
        # Compute the atom description of "atom". Real_bonds need to be
        # symmetric.
        atom_description = np.zeros([1, self.num_atom_descriptors])
        if self.atom_features[0]:
            atom_description[0][0] = self.atoms_type[atom]
        if self.atom_features[1]:
            atom_description[0][1:(self.num_of_types + 1)] \
                = self.get_type_first_nn(atom, real_bonds)
        return atom_description

    def get_type_first_nn(self, atom, real_bonds):
        # Obtain the type (C,H,O,...) of the atoms that are bonded with "atom".
        # Real_bonds need to be symmetric.
        bonded_atom = np.where(real_bonds[atom, :])[0]
        type_first_nn = np.zeros([self.num_of_types])
        for i in bonded_atom:
            type_first_nn[self.atoms_type[i]-1] += 1
        return type_first_nn

    def get_all_atom_features(self, real_bonds):
        # Compute all the atom descriptions of all the atoms in the system.
        # Real_bonds need to be symmetric.
        atoms_descriptions = np.zeros([real_bonds.shape[0], 
                                       self.num_atom_descriptors], dtype=int)
        for atom in range(real_bonds.shape[0]):
            atoms_descriptions[atom] = self.get_atom_features(atom, real_bonds)
        return atoms_descriptions

    def update_all_atom_features(self, real_bonds, bond_change_frame, 
                                 all_atoms_features):
        # Update the atom descriptions after a set of reactions
        # "bond_change_frame" happened. Real_bonds need to be symmetric.
        atoms_involved = np.unique(bond_change_frame[:, 0:2])
        for atom in atoms_involved:
            all_atoms_features[atom] = self.get_atom_features(atom, real_bonds)

