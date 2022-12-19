import numpy as np
import networkx as nx
import pdb

from util import constants

class Molecules:
    ''' Class that tracks the number of molecules through the simulation
    (either analyzing the MD or running the KMC simulation). Each molecule is
    featurized as [# of atom of type 1, # of atom of type 2, ..., # of bonds
    1-1, # of bonds 1-2, ..., of bonds 2-2, ...]. The list of molecules to 
    track can be given in 'molecules_to_track', but if this variable is empty,
    all the molecules are tracked. Other variables such as the size of the 
    longest molecule can be tracked.
    '''

    def __init__(self, reaction_features, num_of_atoms, num_of_types, \
                 atoms_type, atoms, start_frame_MD, end_frame_MD,
                 molecules_to_track, track_extra_variables):
        self.reaction_features = reaction_features
        self.num_of_atoms = num_of_atoms
        self.num_of_types = num_of_types
        self.num_of_molecule_descriptors = self.num_of_types \
                                            + self.num_of_types \
                                            * (self.num_of_types + 1) // 2
        self.atoms_type = atoms_type
        self.atoms = atoms
        self.molecule_list = np.array([], dtype=int)
        self.molecule_list.shape = [0, self.num_of_molecule_descriptors]
        self.track_extra_variables = track_extra_variables
        self.extra_variables = np.zeros([end_frame_MD - start_frame_MD + 1, \
                                         np.sum(self.track_extra_variables)])
        
        # Initialize the graph of the system
        self.molecule_graph = nx.Graph(np.zeros([2, 2]))

        # Initialize the array to track the time
        self.time_range = np.array([])

        # Initialize the array that will track the number of molecules
        if molecules_to_track.shape[0] == 0:
            self.molecule_all_frames = \
                            np.zeros([end_frame_MD - start_frame_MD + 1, \
                                      constants.init_num_of_molecules],
                                      dtype=int)
            self.track_all_molecules = 1
        else:
            self.molecule_all_frames = \
                            np.zeros([end_frame_MD - start_frame_MD + 1,
                                      molecules_to_track.shape[0]],
                                      dtype=int)
            self.set_molecules_to_track(molecules_to_track)
            self.idx_for_mol_to_track = -np.ones([molecules_to_track.shape[0]],
                                                  dtype=int)
            self.all_idx = 0
            self.track_all_molecules = 0

        # Initialize the number of frames with a reaction
        self.num_frames_with_reax = 0

        # Initialize the array that will track the number of molecules at each
        # frame
        self.molecule_per_frame = np.zeros(self.molecule_list.shape[0], \
                                           dtype=int)

    def set_molecules_to_track(self, molecules_to_track):
            try:
                self.molecules_to_track = \
                np.array([i.split('-') for i in molecules_to_track], dtype=int)
            except:
                print('Molecules to track not in the right format')
            assert self.molecules_to_track.shape[1] == \
                                self.num_of_molecule_descriptors, \
                                'Molecules to track not in the right format'

#    def set_molecule_list(self, molecules_to_track):
#        # Get the list of the molecules to track, or if all molecules are
#        # tracked, set track_all_molecules to 1.
#        if molecules_to_track.shape[0] == 0:
#            self.molecule_list = np.array([], dtype=int)
#            self.molecule_list.shape = [0, self.num_of_molecule_descriptors]
#            self.track_all_molecules = 1
#        else:
#            try:
#                self.molecule_list = \
#                np.array([i.split('-') for i in molecules_to_track], dtype=int)
#            except:
#                print('Molecules to track not in the right format')
#            assert self.molecule_list.shape[1] == \
#                                self.num_of_molecule_descriptors, \
#                                'Molecules to track not in the right format'
#            self.track_all_molecules = 0
            
    def initialize_molecules(self, first_frame, start_frame_MD):
        # Initialize what is needed to follow the number of molecules through 
        # the simulation.
        self.create_graph(first_frame)
        self.time_range = np.array([start_frame_MD], dtype=int)
        molecules_frame_list, molecules_frame_counts = \
                                self.get_molecules_frame(self.molecule_graph)
        self.update_molecule_per_frame_and_dict(molecules_frame_list, \
                                                molecules_frame_counts)
        self.update_molecule_all_frames()
        self.num_frames_with_reax = 1

    def create_graph(self, bond_matrix):
        # Create the graph of molecules
        self.molecule_graph = nx.Graph(bond_matrix)
        for atom in range(self.num_of_atoms):
            self.molecule_graph.nodes[atom]['type'] = self.atoms_type[atom]
        for edge in self.molecule_graph.edges:
            self.molecule_graph.edges[edge]['type'] = self.get_edge_type(edge)

    def get_edge_type(self, edge):
        # Get the type of an edge, depending on the type of the atoms it is 
        # bonding. For example, when there are 3 types, this function associates
        # an edge 1-1 to 0, 1-2 to 1, 1-3 to 2, 2-2 to 3, 2-3 to 4, and 3-3 to
        # 5.
        m1 = min(self.molecule_graph.nodes[edge[0]]['type'], \
                 self.molecule_graph.nodes[edge[1]]['type'])
        m2 = max(self.molecule_graph.nodes[edge[0]]['type'], \
                 self.molecule_graph.nodes[edge[1]]['type'])
        return (m1 - 1) * self.num_of_types - m1 * (m1 - 1) // 2 + m2 - 1

    def get_molecules_frame(self, molecule_graph_temp, \
                            atoms_involved=np.array([])):
        # Get the molecules at one specific frame. 

        # Get a generator with the connected components present in the graph.
        connected_components = nx.connected_components(molecule_graph_temp)

        # Initialize the list of the molecules in the frame
        molecules_frame_full_list = \
                np.zeros([nx.number_connected_components(molecule_graph_temp), 
                          self.num_of_molecule_descriptors], dtype=int)

        mol_num = 0

        # For each connected components, save the description as a molecule by
        # obtaining the number of atom of each type and the number of bonds of
        # each type.
        for cc in connected_components:
            cc_graph = molecule_graph_temp.subgraph(cc)

            list_node_type_cc = \
                       list(nx.get_node_attributes(cc_graph, 'type').values())
            types_n, counts_n = np.unique(list_node_type_cc, return_counts=True)
            
            list_edge_type_cc = \
                       list(nx.get_edge_attributes(cc_graph, 'type').values())
            types_e, counts_e = np.unique(list_edge_type_cc, return_counts=True)

            molecules_frame_full_list[mol_num, types_n - 1] = counts_n

            if types_e.shape[0] > 0:
                molecules_frame_full_list[mol_num, types_e + self.num_of_types]\
                                        = counts_e

            mol_num += 1

        molecules_frame_list, molecules_frame_counts = \
                            np.unique(molecules_frame_full_list, \
                                      return_counts=True, axis=0)

        return molecules_frame_list, molecules_frame_counts

    def update_molecule_per_frame_and_dict(self, molecules_frame, 
                                           molecules_count):
        # Given the molecules at a frame and their counts, update 
        # molecule_per_frame and molecule_list. 
        for mol in range(molecules_frame.shape[0]):
            molecule_idx = np.where((self.molecule_list \
                                     == molecules_frame[mol, :]).all(axis=1))

            # Add new molecule if needed
            if molecule_idx[0].shape[0] == 0:
                self.molecule_list = np.vstack([self.molecule_list, \
                                                molecules_frame[mol,:]])
                self.molecule_per_frame = np.append(self.molecule_per_frame, \
                                                    molecules_count[mol])
            elif molecule_idx[0].shape[0] == 1:
                self.molecule_per_frame[molecule_idx[0]] += molecules_count[mol]

    def update_molecule_all_frames(self):
        # Update the list of number of all molecules at all frames.

        self.check_size_molecule_all_frames_and_extra_variables()

        # Update molecule_all_frames
        if self.track_all_molecules == 1:
            self.molecule_all_frames[self.num_frames_with_reax, \
                                     :self.molecule_per_frame.shape[0]] \
                                        = self.molecule_per_frame
        else:
            if self.all_idx == 1:
                self.molecule_all_frames[self.num_frames_with_reax, :] \
                            = self.molecule_per_frame[self.idx_for_mol_to_track]
            else:
                for i in range(self.molecules_to_track.shape[0]):
                    if self.idx_for_mol_to_track[i] != -1:
                        self.molecule_all_frames[self.num_frames_with_reax, i] \
                            = self.molecule_per_frame[\
                                self.idx_for_mol_to_track[i]]
                    else:
                        idx = np.where((self.molecule_list \
                                  == self.molecules_to_track[i]).all(-1))[0]
                        if idx.shape[0] == 1:
                            self.idx_for_mol_to_track[i] = idx[0]
                            self.molecule_all_frames[self.num_frames_with_reax, i] \
                                 = self.molecule_per_frame[\
                                    self.idx_for_mol_to_track[i]]
                            if np.where(self.idx_for_mol_to_track == -1)[0] \
                                    .shape[0] == 0:
                                self.all_idx == 1

        # Update extra_variables if necessary
        self.update_extra_variables()

    def check_size_molecule_all_frames_and_extra_variables(self):
        # As the final number of molecules is not known, molecule_all_frames
        # start with a defined number of molecules. If this number is
        # overtaken, the size of molecule_all_frames is increased.
        if self.molecule_all_frames.shape[1] < self.molecule_list.shape[0] \
            and self.track_all_molecules == 1:
            store_molecule_all_frames = self.molecule_all_frames.copy()
            self.molecule_all_frames = \
                                np.zeros([store_molecule_all_frames.shape[0], \
                                2*store_molecule_all_frames.shape[1]], \
                                dtype=int)
            self.molecule_all_frames[0:store_molecule_all_frames.shape[0], \
                                     0:store_molecule_all_frames.shape[1]] \
                                     = store_molecule_all_frames

        # In the KMC, the final number of molecules is not known, 
        # molecule_all_frames and extra_variables
        # start with a defined number of molecules. If this number is
        # overtaken, ther size of molecule_all_frames is increased.
        if self.num_frames_with_reax == self.molecule_all_frames.shape[0]:
            store_molecule_all_frames = self.molecule_all_frames.copy()
            self.molecule_all_frames \
                = np.zeros([2*store_molecule_all_frames.shape[0],
                            store_molecule_all_frames.shape[1]], dtype=int)
            self.molecule_all_frames[0:store_molecule_all_frames.shape[0], \
                                     0:store_molecule_all_frames.shape[1]] \
                                     = store_molecule_all_frames

            store_extra_variables = self.extra_variables.copy()
            self.extra_variables \
                = np.zeros([2*store_extra_variables.shape[0],
                            store_extra_variables.shape[1]])
            self.extra_variables[0:store_extra_variables.shape[0], \
                                     0:store_extra_variables.shape[1]] \
                                     = store_extra_variables
            

    def update_extra_variables(self):
        # Update the extra variables. In the order they are the number of atom
        # of type 1 in the biggest molecule, the ratio of the number of atoms
        # of type 2 on the number of atoms of type 1 in the biggest molecule.
        if np.sum(self.track_extra_variables) > 0:
            mol_present = np.where(self.molecule_per_frame)

            if self.track_extra_variables[0]:
                biggest_mol = np.max(self.molecule_list[mol_present, 0])
                self.extra_variables[self.num_frames_with_reax, 0] \
                                                                = biggest_mol

            if self.track_extra_variables[1]:
                idx = np.argmax(self.molecule_list[mol_present, 0])
                ratio_1on2 = self.molecule_list[mol_present, 1][0][idx] \
                             / self.molecule_list[mol_present, 0][0][idx]
                self.extra_variables[self.num_frames_with_reax, \
                                     np.sum(self.track_extra_variables[:1])] \
                                     = ratio_1on2 

    def update_after_reaction(self, frame, bond_change_frame):
        # Update the count of molecules when a reaction has occurred.
        self.time_range = np.append(self.time_range, frame)
        molecules_list_change, molecules_counts_change = \
                                self.get_molecules_change(bond_change_frame)
        self.update_molecule_per_frame_and_dict(molecules_list_change, \
                                                molecules_counts_change)
        self.update_molecule_all_frames()
        self.num_frames_with_reax += 1

    def get_molecules_change(self, bond_change):
        # Compute the change in molecules after a reaction, only considering
        # the molecules that contained an atom that reacted.
        atoms_involved = np.unique(bond_change[:, :2])
        molecule_subset_old = set()
        molecule_subset_new = set()
        
        # Get all atoms in molecules containing an atom that reacted before 
        # the reaction occurred.
        for atom in atoms_involved:
            molecule_subset_old = molecule_subset_old.union( \
                        nx.node_connected_component(self.molecule_graph, atom))

        # Get the molecules containing an atom that reacted before the reaction
        # occurred.
        molecule_graph_old = self.molecule_graph.subgraph(molecule_subset_old)
        old_molecules_list, old_molecules_count = self.get_molecules_frame( \
                                            molecule_graph_old, atoms_involved)

        # Make the reactions occur by updating the graph.
        self.update_molecule_graph(bond_change)

        # Get all atoms in molecules containing an atom that reacted after 
        # the reaction occurred.
        for atom in atoms_involved:
            molecule_subset_new = molecule_subset_new.union( \
                        nx.node_connected_component(self.molecule_graph, atom))


        # Get the molecules containing an atom that reacted after the reaction
        # occurred.
        molecule_graph_new = self.molecule_graph.subgraph(molecule_subset_new)
        new_molecules_list, new_molecules_count = self.get_molecules_frame( \
                                            molecule_graph_new, atoms_involved)

        # Get the change in the number of molecules by adding the molecules
        # present after the reaction and removing the molecules that disappear
        # during the reaction.
        molecules_list_change = new_molecules_list.copy()
        molecules_counts_change = new_molecules_count.copy()
        for mol in range(old_molecules_list.shape[0]):
            molecule_idx = np.where((molecules_list_change \
                                    == old_molecules_list[mol, :]).all(axis=1))
            if molecule_idx[0].shape[0] == 0:
                molecules_list_change = np.vstack([molecules_list_change, \
                                                    old_molecules_list[mol, :]])
                molecules_counts_change = np.append(molecules_counts_change, \
                                                    -old_molecules_count[mol])
            elif molecule_idx[0].shape[0] == 1:
                molecules_counts_change[molecule_idx[0]] -= \
                                                    old_molecules_count[mol]
        return molecules_list_change, molecules_counts_change

    def update_molecule_graph(self, bond_change):
        # Update the graph of molecules.
        for i in range(bond_change.shape[0]):
            if bond_change[i, 2] == 1:
                self.molecule_graph.add_edge(bond_change[i, 0], bond_change[i, 1], type=self.get_edge_type((bond_change[i, 0], bond_change[i, 1])))
            if bond_change[i, 2] == 0:
                self.molecule_graph.remove_edge(bond_change[i, 0], bond_change[i, 1])

