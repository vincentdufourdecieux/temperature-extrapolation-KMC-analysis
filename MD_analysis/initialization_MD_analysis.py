import numpy as np
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier
import os

from util import constants

def initialize_system(filename, bond_length_criteria):
    # Get the OVITO pipeline of the simulation, the number of atoms, of 
    # frames, of different types of atoms of the simulation, and the type of
    # each atom. Set the bond length criteria for each type of bond.
    print(os.getcwd())
    pipeline = import_file(constants.md_path + filename)
    data = pipeline.compute()
    num_of_atoms = data.particles.count
    num_of_frames = pipeline.source.num_frames
    num_of_types = np.unique(data.particles.particle_types).shape[0]
    set_bond_length(bond_length_criteria, pipeline, num_of_types)
    atoms_type = get_atoms_type(data, num_of_atoms)
    return pipeline, num_of_atoms, num_of_frames, num_of_types, atoms_type

def set_bond_length(bond_length_criteria, pipeline, num_of_types):
    # Define the bond length criterion for each type of bond
    cbm = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
    bond_type = 0
    for atom_type_1 in range(1, num_of_types + 1):
        for atom_type_2 in range(atom_type_1, num_of_types + 1):
            cbm.set_pairwise_cutoff(atom_type_1, 
                                    atom_type_2,
                                    bond_length_criteria[bond_type])
            bond_type += 1
    pipeline.modifiers.append(cbm)

def get_atoms_type(data, num_of_atoms):
    # Get the type of each atom
    atoms_type = np.zeros([num_of_atoms], dtype=int)
    atoms_type[data.particles['Particle Identifier'] - 1] = \
                                            data.particles.particle_types[:]
    return atoms_type

