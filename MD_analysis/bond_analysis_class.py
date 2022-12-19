import numpy as np
import pdb
from scipy import sparse
import time

from util import constants

class BondAnalysis:
    ''' Class that will apply the bond length and time criteria to know which
    atoms are bonded and when in the MD simulation. When a bond changes of
    status (broken or not), a reaction happens and the list of the different
    reactions and their descriptions is recorded.
    '''

    def __init__(self, pipeline, bond_duration, num_of_atoms, reactions, \
                 molecules, start_frame_MD, end_frame_MD, \
                 start_frame_analysis, end_frame_analysis):
        self.pipeline = pipeline
        self.bond_duration = bond_duration
        self.num_of_atoms = num_of_atoms
        self.reactions = reactions
        self.molecules = molecules
        self.start_frame_MD = start_frame_MD
        self.end_frame_MD = end_frame_MD
        self.start_frame_analysis = start_frame_analysis
        self.end_frame_analysis = end_frame_analysis
        
        # Initialize real_bonds that is the adjacency matrix of the system
        # considering the bond length and time criteria.
        self.real_bonds = sparse.dok_matrix((self.num_of_atoms, \
                                             self.num_of_atoms))

        # Initialize bond_follow that will count for how long a bond has changed
        # status. When the time a bond changed status reaches the bond_duration
        # parameter, the bond is changed in self.real_bonds and a reaction is 
        # considered to have occurred.
        self.bond_follow = sparse.csr_matrix((self.num_of_atoms, \
                                              self.num_of_atoms))

        # Initialize bond_change that is a dictionary keeping track of the 
        # change of status of the bonds. The keys will be the frames at which
        # at least one bond has changed status and the values will be the 
        # description of this change of status. The description of the change
        # of status will be a list:
        # [index of atom 1, index of 2, 0 if bond is broken and 1 if created].
        self.bond_change = {}

        # Initialize volumes that will track the volume of the MD cell through
        # the simulation.
        self.volumes = np.zeros([self.end_frame_MD - self.start_frame_MD])

    def run_bond_analysis_and_extract_reactions(self):
        # Run the molecular analyzer, for each frame, check if a reaction 
        # happens using the bond criteria. If yes, store the reactions, the 
        # number of times it happened, its type... Keep the first frame and the
        # bond changes.

        frame_counter = constants.frame_bond_analysis_percentage_counter

        for frame in range(self.start_frame_MD, self.end_frame_MD):
            # Print the evolution of the analysis
            if frame/(self.end_frame_MD - self.start_frame_MD) > frame_counter:
                print(frame_counter, 
                      flush=True)
                print(time.process_time(), flush=True)
                frame_counter += \
                    constants.frame_bond_analysis_percentage_counter_increment

            data = self.pipeline.compute(frame)
            self.volumes[frame] = data.cell.volume
            
            bond_change_frame = self.apply_bond_duration_criterion(frame, data)


            # If bond changes occur, featurize reaction and save it and update
            # the molecules count.
            if bond_change_frame.shape[0] != 0:
                bond_change_frame \
                        = bond_change_frame[bond_change_frame[:, 0].argsort()]

                if frame >= self.start_frame_MD + 2*self.bond_duration + 1:
                    corrected_frame = frame - 2*self.bond_duration
                    self.bond_change[corrected_frame] = bond_change_frame
                    self.molecules.update_after_reaction(corrected_frame, \
                                            self.bond_change[corrected_frame])

                if self.start_frame_analysis + 2*self.bond_duration + 1 \
                   <= frame <= self.end_frame_analysis:
                    self.reactions.update_reactions_and_real_bonds( \
                            bond_change_frame, self.real_bonds, \
                            frame - 2*self.bond_duration)

                # Update real_bonds
                for reax in range(bond_change_frame.shape[0]):
                    self.real_bonds[bond_change_frame[reax, 0], \
                       bond_change_frame[reax, 1]] = bond_change_frame[reax, 2]

            # When frame is equal to start_frame_MD + the number of time frames
            # that allow not to have initialization effects, store first_frame
            # and start computing the number of molecules.
            if frame == self.start_frame_MD + 2*self.bond_duration:
                first_frame = self.real_bonds.copy()
                first_frame_symmetric = (first_frame.T + first_frame).toarray()
                self.molecules.initialize_molecules(first_frame_symmetric, \
                                                    frame)

            # When frame is equal to start_frame_analysis + the number of time
            # frames that allow not to have initialization effects, store
            # first_frame_analysis.
            if frame == self.start_frame_analysis + 2*self.bond_duration:
                first_frame_analysis = self.real_bonds.copy()

        volume = np.mean(self.volumes)
        return first_frame, self.bond_change, first_frame_analysis, volume

    def apply_bond_duration_criterion(self, frame, data):
        # Apply the bond duration criteria. Check if a bond has been broken or 
        # created for "bond_duration" number of steps, otherwise put back the 
        # count to 0. If a reaction happens, store it in bond_change_frame.
        
        bonds_frame = self.get_bonds_frame(data)

        # Array that will track the descriptions of the bond changes occurring
        # during this frame.
        bond_change_frame = np.array([], dtype=int)
        bond_change_frame.shape = [0, 3]

        # Update bond_follow
        if frame == self.start_frame_MD:
            self.real_bonds = bonds_frame
        else:
            abs_difference = abs(self.real_bonds - bonds_frame)
            self.bond_follow = self.bond_follow.multiply(abs_difference) \
                               + abs_difference

        # Find if bond changes occurred and store them
        bond_change_occurring = \
                            sparse.find(self.bond_follow == self.bond_duration)
        if bond_change_occurring[0].shape[0] != 0:
            for reax in range(bond_change_occurring[0].shape[0]):
                atom_1 = bond_change_occurring[0][reax]
                atom_2 = bond_change_occurring[1][reax]
                bond_change_description = \
                    [[atom_1, atom_2, \
                    int(1 - self.real_bonds[atom_1, atom_2])]]
                bond_change_frame = \
                        np.append(bond_change_frame, bond_change_description, 0)
                self.bond_follow[atom_1, atom_2] = 0

        return bond_change_frame

    def get_bonds_frame(self, data):
        # Compute bonds_frame that is the adjacency matrix of the system at 
        # this frame.
        bonds_frame = sparse.dok_matrix((self.num_of_atoms, self.num_of_atoms))

        # Extract OVITO's list of bonds and converter from particles index to
        # particles identifiers.
        bonds_list_indexes = data.particles.bonds.topology
        particles_indexes_to_identifiers = data.particles.identifiers

        # The list of the bonds is given by OVITO with the indexes of the
        # particles which change at each frame. So it is converted to the 
        # identifiers of the particles that are always the same.
        bonds_list_identifiers = \
                particles_indexes_to_identifiers[bonds_list_indexes] - 1

        # The lowest identifiers is put first so that bonds_frame is an upper
        # triangular array.
        bonds_list_identifiers.sort()
        bonds_frame[bonds_list_identifiers[:, 0], \
                                    bonds_list_identifiers[:, 1]] = 1
        return bonds_frame
