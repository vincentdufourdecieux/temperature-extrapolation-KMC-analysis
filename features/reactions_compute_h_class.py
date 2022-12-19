import itertools
import networkx as nx
import numpy as np
import pdb
import time

from util import constants


class Reactions:
    ''' Class defining the reactions descriptions. Here reactions can involve
    2 or 3 atoms. A reaction is featibed by:
    - the number of atoms involved
    - the features of the atoms involved sorted by lexicographic order. So
    [1, 1, 3] will come before [2, 1, 0] which will come before [2, 2, 0].
    - the features of the bonds that are evolving. This feature will be
    the index of the first atom involved, the index of the second atom involved
    and 0 or 1 if the bond between the two atoms is broken or created, 
    respectively.
    
    For example, the reaction involving 3 atoms with features [1, 3, 1], 
    [1, 1, 3], and [2, 1, 0], where the bond between [1, 1, 3] and [2, 1, 0] is
    broken and the bond between [1, 3, 1] and [2, 1, 0] is created will have 
    a feature:
    [3, 1, 1, 3, 1, 3, 1, 2, 1, 0, 0, 2, 0, 1, 2, 1].
    '''
    def __init__(self, atom_features, reaction_features, num_of_types, \
                max_atoms_in_reactions, atoms, molecules, reaction_dict, \
                clusters_to_follow, clusters_in_reactions, \
                atom_features_in_reactions, \
                atom_features_in_clusters_to_follow, MD_or_KMC):

        self.atom_features = atom_features
        self.reaction_features = reaction_features
        self.num_atom_features = self.atom_features[0] \
                                    + num_of_types * self.atom_features[1]
        self.all_atom_features = np.array([], dtype=int)

        # Set the maximum number of atoms that can involved in one reaction.
        self.max_atoms_in_reactions = max_atoms_in_reactions

        self.atoms = atoms
        self.molecules = molecules

        # Initialize (for MD analysis) or retrieve (for KMC simulation) the 
        # different features of each reaction

        # Dictionary containing the feature of the different reactions. 
        # The keys are the index of the reaction, the values are an array 
        # featibing the reactions.
        self.reaction_dict = reaction_dict 

        # Dictionary containing the feature of a cluster of atoms that needs
        # to be followed as they appear in at least one reaction. These clusters
        # can contain between 1 and 3 atoms. Their feature is:
        # - number of atoms in the cluster (1 to 3)
        # - atom features of each of the atoms.
        # - bonds in the cluster: none for 1 atom, always 1 for 2 atoms, 
        # [binary if there is a bond between 1st and 2nd atom, 1st and 3rd atom,
        # 2nd and 3rd atom] for 3 atoms.
        # The keys are the index of the clusters to be followed,
        # the values are an array of their feature. 
        self.clusters_to_follow = clusters_to_follow

        # Dictionary containing the clusters that are present in each reaction,
        # and the corresponding atoms in the reaction_dict feature. This is
        # contained in a list of 2 elements. The first element is a list
        # containing the indexes of the clusters in the reaction. The second
        # element is a list that follows the order of the first element and 
        # gives in an array which atoms (in the order of reaction_dict) the
        # clusters contain. For example if the reaction 3 involves cluster 1 and
        # cluster 3 and that cluster 1 contains the first and second atom of
        # the feature of reaction 3 (in reaction_dict) and that cluster 3
        # contains the third atom, then clusters_in_reactions[3] will be equal
        # to : [[1, 3], [array([0, 1]), array([2])].
        self.clusters_in_reactions = clusters_in_reactions

        # Dictionary containing the atom features of the atoms involved in
        # the reaction. The keys are the reactions index and the values are a 
        # list of the features of the atoms involved.
        self.atom_features_in_reactions = atom_features_in_reactions

        # Dictionary containing the atom features of the atoms contained in
        # the clusters. The keys are the clusters index and the values are a
        # list of the features of the atoms contained in the cluster.
        self.atom_features_in_clusters_to_follow = \
            atom_features_in_clusters_to_follow


        # Dictionary containing the reactions that contain a specific atom 
        # feature. The keys are the atom features and the values are
        # a list of the reactions indexes.
        self.reactions_per_atom_features = \
            self.get_reactions_per_atom_features()

        # Dictionary containing the clusters that contain a specific atom 
        # feature. The keys are the atom features and the values are
        # a list of the clusters indexes.
        self.clusters_to_follow_per_atom_features = \
            self.get_clusters_to_follow_per_atom_features()

        # Dictionary containing a dictionary for each cluster. The dictionary
        # associated with each cluster contains the other atom of the cluster
        # to which a specific atom of the cluster is bonded. The keys of the
        # dictionary associated with each cluster are the indexes of each atom
        # (generally 0, 1, and 2) in the cluster. And the values are a list
        # of the other atoms to which this specific atom is bonded in the 
        # cluster.
        self.bonds_in_clusters_to_follow = \
            self.get_bonds_in_clusters_to_follow()

        self.list_of_clusters_to_follow = {}
        self.list_of_reactants = {}

        # 0 if MD analysis, 1 if KMC simulation
        self.MD_or_KMC = MD_or_KMC


    def get_reactions_per_atom_features(self):
        reactions_per_atom_features = {}
        for r in self.atom_features_in_reactions.keys():
            list_feat = self.atom_features_in_reactions[r]
            for num_feat in range(list_feat.shape[0]):
               feat = str(list_feat[num_feat])
               if feat in reactions_per_atom_features.keys():
                    reactions_per_atom_features[feat].append(r)
               else:
                    reactions_per_atom_features[feat] = [r]
        return reactions_per_atom_features

    def get_clusters_to_follow_per_atom_features(self):
        clusters_to_follow_per_atom_features = {}
        for c in self.atom_features_in_clusters_to_follow.keys():
            list_feat = self.atom_features_in_clusters_to_follow[c]
            for num_feat in range(list_feat.shape[0]):
               feat = str(list_feat[num_feat])
               if feat in clusters_to_follow_per_atom_features.keys():
                    clusters_to_follow_per_atom_features[feat].append(c)
               else:
                    clusters_to_follow_per_atom_features[feat] = [c]
        return clusters_to_follow_per_atom_features

    def get_bonds_in_clusters_to_follow(self):
        # Get the order of the bonds. For example, for 3 atoms in the reaction
        # the first bond is atom 0-atom 1, the second is atom 0-atom 2, and the
        # third one is atom 1-atom 2
        bonds_in_clusters_to_follow = {}
        for c in self.clusters_to_follow.keys():
            clus_feat = self.clusters_to_follow[c]
            num_of_atom = clus_feat[0]
            if num_of_atom == 1:
                bonds_in_clusters_to_follow[c] = {}
                continue
            num_to_idx = self.get_num_to_idx(num_of_atom)
    
            # For each pair of atom, get if they are connected or not.
            bonds_start_idx = 1 + num_of_atom * self.num_atom_features
            connections = np.hstack(\
                (num_to_idx, 
                 self.clusters_to_follow[c][bonds_start_idx:].reshape(-1, 1)))
    
            # Remove the pairs of atoms that are not connected.
            connections = np.delete(connections[:, :2], 
                                    np.where(connections[:, 2] == 0)[0], axis=0)
    
            dict_of_bonds = {}
            for atom in range(num_of_atom):
                dict_of_bonds[atom] = []
                for bond in range(connections.shape[0]):
                    if atom == connections[bond, 0]:
                        dict_of_bonds[atom].append(connections[bond,1])
                    elif atom == connections[bond, 1]:
                        dict_of_bonds[atom].append(connections[bond, 0])
            bonds_in_clusters_to_follow[c] = dict_of_bonds
        return bonds_in_clusters_to_follow

    def get_h(self, first_frame, bond_change, start_frame, end_frame):
        # Get the number of times each reaction could have happened for each 
        # frame between start_frame and end_frame.

        # Initialize the h arrays.
        h_tot = np.zeros(len(self.reaction_dict), dtype=int)
        h_per_frame = np.zeros([len(self.reaction_dict), 
                                end_frame - start_frame], dtype=int)
        bond_matrix = first_frame.copy()

        # Get the features of all the atoms
        self.all_atom_features \
            = self.atoms.get_all_atom_features(bond_matrix)

        # For each cluster_to_follow, obtain all the combinations of atoms
        # that form the correct cluster_to_follow
        self.get_list_of_clusters_to_follow(bond_matrix)

        h_prev = self.get_h_frame(bond_matrix)
        h_per_frame[:, 0] = h_prev

        frame_counter = constants.get_h_percentage_counter

        for frame in range(start_frame, end_frame):

            if frame/(end_frame - start_frame) > frame_counter:
                print(frame_counter, flush=True)
                print(time.process_time(), flush=True)
                frame_counter += constants.get_h_percentage_counter

            h_tot += h_prev
            h_per_frame[:, frame - start_frame] = h_prev

            if frame in bond_change.keys():
                # Update h after the reaction happened
                h_prev = self.update_all_after_reax(h_prev, bond_matrix, 
                                                    bond_change[frame])
        return h_tot, h_per_frame

    def get_list_of_clusters_to_follow(self, bond_matrix):
        # Get the number of all the clusters_to_follow given the bond_matrix 
        # (adjacency matrix) and the features of the atoms.
        for cluster in range(len(self.clusters_to_follow)):
            num_of_atom = self.clusters_to_follow[cluster][0]

            # Case where there is only one atom in the cluster
            if num_of_atom == 1:
                self.update_list_of_reactant_one_atom(cluster)

            # Case where there are several atoms in the cluster
            else:
                # Initialize the array that will store all the possible 
                # combinations of atoms that form one cluster with the good
                # feature
                num_to_idx = self.get_num_to_idx(num_of_atom)
                possible_comb = np.array([], dtype=int)
                possible_comb.shape = [0, num_of_atom]
                visited = 0

                # Consider each possible bond in the cluster
                for i in range(len(num_to_idx)):
                    # Only the atoms that are bonded define the cluster, the
                    # other atoms can be bonded or not
                    if self.clusters_to_follow[cluster]\
                            [1 + num_of_atom*self.num_atom_features + i] \
                                == 1:
                        # Get the index, in the order of the cluster, of the 
                        # atoms that are consider
                        idx_1 = num_to_idx[i][0]
                        idx_2 = num_to_idx[i][1]

                        # Get the feature of the atoms
                        atom_1_feature = self.clusters_to_follow[cluster]\
                            [1 + idx_1 * self.num_atom_features: \
                                1 + (idx_1 + 1) * self.num_atom_features]
                        atom_2_feature = self.clusters_to_follow[cluster]\
                            [1 + idx_2 * self.num_atom_features: \
                                1 + (idx_2 + 1) * self.num_atom_features]

                        if visited == 0:
                            # If it is the first bond of the cluster, 
                            # initialize all the possible bonds between the 
                            # atoms of the correct features.
                            possible_comb = self.initialize_possible_comb( \
                                possible_comb, atom_1_feature, 
                                atom_2_feature, bond_matrix, idx_1, idx_2, 
                                num_of_atom)
                            visited = 1
                        else:
                            # If it is not the first bond of the cluster, then
                            # possible_comb is not empty. Therefore, there are
                            # three cases, either the two atoms that are being
                            # considered now where not considered in the 
                            # previous bond, and so the columns in possible_comb
                            # of both of these atoms will be -1, or one of the
                            # atoms will already be considered and the columns
                            # of the possible_comb of this atom won't be -1, 
                            # or both of the atoms will already be considered.
                            # Between these two cases, the way to update 
                            # possible_comb will be different.
                            if possible_comb[0, idx_1] != -1 \
                                and possible_comb[0, idx_2] != -1:
                                possible_comb \
                                    = self.update_possible_comb_2atoms_present(\
                                        possible_comb, bond_matrix, idx_1, 
                                        idx_2, num_of_atom)
                            elif possible_comb[0, idx_1] != -1 \
                                and possible_comb[0, idx_2] == -1:
                                possible_comb \
                                    = self.update_possible_comb_1atom_present(
                                        possible_comb, bond_matrix, idx_1, 
                                        idx_2, num_of_atom, atom_2_feature)
                            elif possible_comb[0, idx_1] == -1 \
                                and possible_comb[0, idx_2] != -1:
                                possible_comb = \
                                    self.update_possible_comb_1atom_present(
                                        possible_comb, bond_matrix, idx_2, 
                                        idx_1, num_of_atom, atom_1_feature)
                            else:
                                possible_comb \
                                    = self.update_possible_comb_noatom_present(\
                                        possible_comb, atom_1_feature, 
                                        atom_2_feature, bond_matrix, idx_1,
                                        idx_2, num_of_atom)
                    
                    # If two atoms of the cluster have the same feature,
                    # the cases where the same two atoms are in one of the
                    # possible combinations must be removed.
                    if np.array_equal(atom_1_feature, atom_2_feature):
                        row_with_same = np.argwhere(possible_comb[:, idx_1] 
                                                    == possible_comb[:, idx_2])
                        possible_comb = np.delete(possible_comb, row_with_same, 
                                                  axis=0)

                    # If one of the bond has considered and there are no
                    # possible combination, it is not necessary to consider
                    # the other bonds as the cluster won't be observed anyway.
                    if visited == 1 and possible_comb.shape[0] == 0:
                        self.list_of_clusters_to_follow[cluster] = np.array([])
                        break

                # Update list_of_clusters_to_follow with all the possible 
                # combinations of the cluster that are present in the system
                # at this time.
                self.list_of_clusters_to_follow[cluster] = possible_comb

            # Remove rows with duplicates.
            if self.list_of_clusters_to_follow[cluster].shape[0] > 0:
                self.list_of_clusters_to_follow[cluster] \
                    = self.rows_uniq_elems1( \
                        self.list_of_clusters_to_follow[cluster])

    def update_list_of_reactant_one_atom(self, cluster):
        # When there is only one atom in the cluster, the list of clusters with
        # the correct feature can be obtained from all_atom_features.
        indexes = np.argwhere((self.all_atom_features \
             == self.clusters_to_follow[cluster][ \
                1: 1 + self.num_atom_features]).all(-1))
        if indexes.shape[0] == 0:
            self.list_of_clusters_to_follow[cluster] = np.array([])
        else:
            self.list_of_clusters_to_follow[cluster] \
                = np.reshape(np.concatenate(indexes), (-1, 1))

    def get_num_to_idx(self, num_of_atom):
        # In the cluster feature, the order of the status of the bonds are
        # between atom 0 and 1, between atom 0 and 2,... between atom 1 and 2,
        # between atom 1 and 3...
        num_to_idx = []
        for i in range(num_of_atom - 1):
            for j in range(i + 1, num_of_atom):
                num_to_idx.append([i, j])
        return num_to_idx

    def initialize_possible_comb(self, possible_comb, atom_1_feature, 
                                 atom_2_feature, bond_matrix, idx_1, idx_2,
                                 num_of_atom):
        # To initialize all the possible combinations of the cluster, we start
        # by getting all the atoms that have the same feature as the first
        # atom of the bond considered. Then, we check if the atoms they are 
        # bonded have the same feature as the second atom of the bond
        # considered.
        indexes_atom_1_feature \
            = np.argwhere((self.all_atom_features == atom_1_feature)\
                          .all(-1))
        for atom_1 in indexes_atom_1_feature:
            bonded_atom_1 = np.where(bond_matrix[atom_1[0], :])[0]
            for ba in bonded_atom_1:
                if np.array_equal(atom_2_feature, 
                                  self.all_atom_features[ba]):
                    comb = (-1) * np.ones([num_of_atom], dtype=int)
                    comb[idx_1] = atom_1
                    comb[idx_2] = ba
                    possible_comb = np.vstack((possible_comb, comb.T))
        return possible_comb

    def update_possible_comb_2atoms_present(self, possible_comb, bond_matrix,
                                            idx_1, idx_2, num_of_atom):
        # If the two atoms of the considered bond in the cluster have already
        # been searched, then we just want to check if the indexes of the atoms
        # in possible combinations are actually bonded.
        new_possible_comb = np.array([], dtype=int)
        new_possible_comb.shape = [0, num_of_atom]
        for j in range(possible_comb.shape[0]):
            if bond_matrix[possible_comb[j, idx_1], possible_comb[j, idx_2]] \
                == 1:
                new_possible_comb \
                    = np.vstack([new_possible_comb, possible_comb[j, :]])
        return new_possible_comb

    def update_possible_comb_1atom_present(self, possible_comb, bond_matrix, 
                                           idx_present, idx_not_present, 
                                           num_of_atom, 
                                           atom_feature_not_present):
        # If one of the atoms of the considered bond in the cluster has already
        # been searched, then we just need to check if the atoms that are in the
        # possible combinations are bonded to atoms that have the second
        # feature than the bond.
        new_possible_comb = np.array([], dtype=int)
        new_possible_comb.shape = [0, num_of_atom]
        for j in range(possible_comb.shape[0]):
            bonded_atom_1 \
                = np.where(bond_matrix[possible_comb[j, idx_present], :])[0]
            for ba in bonded_atom_1:
                if np.array_equal(atom_feature_not_present, 
                                  self.all_atom_features[ba]):
                    comb = possible_comb[j, :]
                    comb[idx_not_present] = ba
                    new_possible_comb = np.vstack((new_possible_comb, comb))
        return new_possible_comb

    def update_possible_comb_noatom_present(self, possible_comb, 
                                            atom_1_feature, 
                                            atom_2_feature, bond_matrix, 
                                            idx_1, idx_2, num_of_atom):
        # If no atom are present, then we need to add all possibilities for 
        # the part of the cluster already considered to all the possibilities
        # of the bond considered now.
        new_possible_comb = np.array([], dtype=int)
        new_possible_comb.shape = [0, num_of_atom]
        other_possible_comb = np.array([], dtype=int)
        other_possible_comb.shape = [0, num_of_atom]
        other_possible_comb \
            = self.initialize_possible_comb(other_possible_comb, 
                                            atom_1_feature,
                                            atom_2_feature, bond_matrix, 
                                            idx_1, idx_2, num_of_atom)
        for j in possible_comb.shape[0]:
            for k in other_possible_comb.shape[0]:
                comb = possible_comb[j, :]
                comb[idx_1] = other_possible_comb[k, idx_1]
                comb[idx_2] = other_possible_comb[k, idx_2]
                new_possible_comb = np.vstack((new_possible_comb, comb))
        return new_possible_comb

    def rows_uniq_elems1(self, a):
        # Remove rows where there are duplicates
        idx = a.argsort(1)
        a_sorted = a[np.arange(idx.shape[0])[:, None], idx]
        return a[(a_sorted[:, 1:] != a_sorted[:, :-1]).all(-1)]

    def get_h_frame(self, bond_matrix):
        # We get the number of times h each reaction could have happened.
        num_of_reax = len(self.reaction_dict)
        h = np.zeros([num_of_reax], dtype=int)

        # If we are the first step of the KMC, molecules and 
        # all_atom_features, and list_of_clusters_to_follow are not 
        # initialize yet.
        if self.MD_or_KMC == 1:
            self.molecules.initialize_molecules(bond_matrix, 0)
        if self.all_atom_features.shape[0] == 0:
            self.all_atom_features \
                = self.atoms.get_all_atom_features(bond_matrix)
        if len(self.list_of_clusters_to_follow) == 0:
            self.get_list_of_clusters_to_follow(bond_matrix)

        # For each reaction build the list of reactants, first by doing all
        # the possible combinations of the clusters in the reactions, then 
        # by removing the combinations that don't work as for example two 
        # atoms that should create a bond are already bonded.
        for reax in range(num_of_reax):
            num_of_atoms_in_reax = self.reaction_dict[reax][0]

            all_combi = []
            size_all_combi = 1
            for c in range(len(self.clusters_in_reactions[reax][0])):
                cluster_number = self.clusters_in_reactions[reax][0][c]
                num_cluster_present \
                    = self.list_of_clusters_to_follow[cluster_number].shape[0]
                all_combi.append(range(num_cluster_present))
                size_all_combi *= num_cluster_present

            self.list_of_reactants[reax] \
                = self.create_list_of_reactants(all_combi, reax, 
                                                size_all_combi, 
                                                num_of_atoms_in_reax)

            if self.list_of_reactants[reax].shape[0] != 0:
                self.list_of_reactants[reax] = self.correct_list_of_reactants(\
                                                reax, num_of_atoms_in_reax,
                                                self.list_of_reactants,
                                                bond_matrix)
            h[reax] = self.list_of_reactants[reax].shape[0]

        return h

    def create_list_of_reactants(self, all_combi, reax, size_all_combi, 
                                 num_of_atoms_in_reax):
        # Create the list of all possible combinations of reactants for the 
        # reaction in list_of_reactants[reax]
        list_of_reactants_reax \
            = (-1) * np.ones([size_all_combi, num_of_atoms_in_reax], dtype=int)

        for num_i, i in enumerate(itertools.product(*all_combi)):
            for j in range(len(i)):
                cluster = self.clusters_in_reactions[reax]
                list_of_reactants_reax[num_i, cluster[1][j]] \
                    = self.list_of_clusters_to_follow[cluster[0][j]][i[j]]

        # Remove rows with duplicates
        list_of_reactants_reax = self.rows_uniq_elems1(list_of_reactants_reax)
        return list_of_reactants_reax

    def correct_list_of_reactants(self, reax, num_of_atoms_in_reax, 
                                  list_of_reactants, bond_matrix):
        # Remove the reactions that are not possible, counted several times...
        list_of_reactants_reax = list_of_reactants[reax].copy()
        num_of_bond_change = (self.reaction_dict[reax].shape[0] \
                              - num_of_atoms_in_reax \
                              * self.num_atom_features - 1) // 3
        for i in range(num_of_bond_change):
            start_idx = 1 + num_of_atoms_in_reax * self.num_atom_features \
                        + i * 3
            bond_change = self.reaction_dict[reax][start_idx: start_idx + 3]

            # Check that two atoms that could form a bond are not already bonded
            if bond_change[2] == 1:
                reactant_1 = list_of_reactants_reax[:, bond_change[0]]
                reactant_2 = list_of_reactants_reax[:, bond_change[1]]
                idx_already_bonded \
                    = np.where(bond_matrix[reactant_1, reactant_2])[0]
                list_of_reactants_reax = np.delete(list_of_reactants_reax, 
                                                   idx_already_bonded, axis=0)
        
        return list_of_reactants_reax

    def update_all_after_reax(self, h_prev, bond_matrix, bond_change, frame=-1):
        # Update h, clusters_to_follow... using only the atoms involved in the 
        # reaction and their nearest neighbors. It saves a lot of time to
        # consider only those.
        h = np.zeros(h_prev.shape, dtype=int)
        atoms_involved = np.unique(bond_change[:, :2])

        # If we are in the KMC we need to update the molecules
        if self.MD_or_KMC == 1:
            self.molecules.update_after_reaction(frame, bond_change)

        self.remove_from_list_of_clusters_to_follow(atoms_involved)
        self.remove_from_list_of_reactants(atoms_involved)
        self.update_bond_matrix(bond_matrix, bond_change)

        # Update after so we count the reactants before the reactions happen.
        self.atoms.update_all_atom_features(bond_matrix, bond_change, 
                                            self.all_atom_features)  

        idx_temp_to_real = self.add_to_list_of_clusters_to_follow(\
                            atoms_involved, bond_matrix)
        self.add_to_list_of_reactants(bond_matrix, idx_temp_to_real)
        for reax in range(len(self.list_of_reactants)):
            h[reax] = self.list_of_reactants[reax].shape[0]
        return h

    def remove_from_list_of_clusters_to_follow(self, atoms_involved):
        # In list_of_clusters_to_follow, the clusters containing the atoms that
        # just reacted are removed, as their feature changes with the
        # reaction.
        for atom in atoms_involved:
            feat = str(self.all_atom_features[atom])
            list_cluster = self.clusters_to_follow_per_atom_features[feat]
            for c in list_cluster:
                rows_with_atom = np.where((self.list_of_clusters_to_follow[c]\
                                           == atom).any(-1))[0]
                self.list_of_clusters_to_follow[c]  \
                    = np.delete(self.list_of_clusters_to_follow[c], 
                                rows_with_atom, axis=0)

    def remove_from_list_of_reactants(self, atoms_involved):
        # In list_of_reactants, the reactants containing the atoms that
        # just reacted are removed, as their feature changes with the
        # reaction.
        for atom in atoms_involved:
            feat = str(self.all_atom_features[atom])
            list_reactions = self.reactions_per_atom_features[feat]
            for reax in list_reactions:
                rows_with_atom = np.where((self.list_of_reactants[reax]
                                           == atom).any(-1))[0]
                self.list_of_reactants[reax] = \
                    np.delete(self.list_of_reactants[reax], rows_with_atom,
                              axis=0)

    def update_bond_matrix(self, bond_matrix, bond_change):
        # Update the adjacency matrix after some bond changes.
        for reax in range(bond_change.shape[0]):
            bond_matrix[bond_change[reax][0], bond_change[reax][1]] = \
                bond_change[reax][2]
            bond_matrix[bond_change[reax][1], bond_change[reax][0]] = \
                bond_change[reax][2]

    def add_to_list_of_clusters_to_follow(self, atoms_involved, bond_matrix):
        atoms_involved_in_clusters_to_follow = {}
        list_of_clusters_to_follow_temp = {}
        idx_temp_to_real = {}
        # For each atom, get the list of clusters to follow it is involved in
        # and create a list_of_clusters_to_follow_temp with the clusters
        # that will be used.
#        for atom in atoms_involved:
#            atoms_involved_in_clusters_to_follow[atom] = []
#            for reax in range(len(self.atom_features_in_clusters_to_follow)):
#                if ((self.all_atom_features[atom] == self.atom_features_in_clusters_to_follow[reax]).all(-1)).any(-1):
#                    atoms_involved_in_clusters_to_follow[atom].append(reax)
#                    num_of_atom = self.clusters_to_follow[reax][0]
#                    list_of_clusters_to_follow_temp[reax] = np.array([], dtype=int)
#                    list_of_clusters_to_follow_temp[reax].shape = [0, num_of_atom]
        
        for atom in atoms_involved:
            feat = str(self.all_atom_features[atom])
            if feat not in self.clusters_to_follow_per_atom_features.keys():
                continue
            list_clusters = self.clusters_to_follow_per_atom_features[feat]
            for c in list_clusters:
                num_of_atom = self.clusters_to_follow[c][0]
                if c not in list_of_clusters_to_follow_temp.keys():
                    list_of_clusters_to_follow_temp[c] = np.array([], dtype=int)
                    list_of_clusters_to_follow_temp[c].shape = [0, num_of_atom]
                if num_of_atom == 1:
                    list_of_clusters_to_follow_temp[c] = \
                        np.append(list_of_clusters_to_follow_temp[c], atom)
                else:
                    # In one cluster_to_follow, gives the list of other atoms
                    # to which each atom are bonded.
                    dict_of_bonds = self.bonds_in_clusters_to_follow[c]

                    # Get the list of the atoms features in the cluster
                    c_feat = self.clusters_to_follow[c]
                    atoms_feat_in_c = \
                        c_feat[1 : 1 + self.num_atom_features*num_of_atom]
                    features = atoms_feat_in_c.reshape((-1, 
                                                    self.num_atom_features))
#                    features = np.vstack(\
#                        [self.clusters_to_follow[c][1+ self.num_atom_features*i: 1 + self.num_atom_features*(i+1)]for i in range(num_of_atom)])

                    # In the clusters, see where the atom considered is. For
                    # example, if the cluster is [3, 1, 1, 2, 2, 0, 1, 2, 0, 1,...]
                    # and the atom features are [2, 0, 1] then we'll have
                    # [1, 2].
                    atom_in_clusters_to_follow = \
                        self.get_possible_positions(atom, c_feat)

                    for atom_position in atom_in_clusters_to_follow:
                        new_clusters_to_follow = \
                            self.search_clusters_to_follow(\
                                np.array([]), dict_of_bonds, -1, atom_position,
                                atom, features, bond_matrix)

                        if new_clusters_to_follow.shape[0] > 0:
                            list_of_clusters_to_follow_temp[c] = \
                                np.vstack((\
                                    list_of_clusters_to_follow_temp[c], 
                                    new_clusters_to_follow))

        for c in list_of_clusters_to_follow_temp.keys():
            if list_of_clusters_to_follow_temp[c].shape[0] == 0:
                continue

            if len(list_of_clusters_to_follow_temp[c].shape) == 1:
                list_of_clusters_to_follow_temp[c] = \
                    np.unique(list_of_clusters_to_follow_temp[c])
            else:
                list_of_clusters_to_follow_temp[c] = \
                    self.rows_uniq_elems1(list_of_clusters_to_follow_temp[c])
                if list_of_clusters_to_follow_temp[c].shape[0] > 0:
                    list_of_clusters_to_follow_temp[c] = \
                        np.unique(list_of_clusters_to_follow_temp[c], axis=0)
            if c not in idx_temp_to_real.keys():
                idx_temp_to_real[c] = np.array([], dtype=int)
            idx_temp_to_real[c]= np.append(
                        idx_temp_to_real[c], 
                        list(range(self.list_of_clusters_to_follow[c].shape[0],
                                   self.list_of_clusters_to_follow[c].shape[0]\
                                   + list_of_clusters_to_follow_temp[c].shape[0]\
                                   )))

            if self.list_of_clusters_to_follow[c].shape[0] == 0:
                if len(list_of_clusters_to_follow_temp[c].shape) == 1:
                    self.list_of_clusters_to_follow[c] = \
                        list_of_clusters_to_follow_temp[c].reshape(-1, 1)
                else:
                    self.list_of_clusters_to_follow[c] = \
                        list_of_clusters_to_follow_temp[c]
            elif self.list_of_clusters_to_follow[c].shape[1] == 1:
                self.list_of_clusters_to_follow[c] = np.vstack(
                    (self.list_of_clusters_to_follow[c], 
                     list_of_clusters_to_follow_temp[c].reshape(-1, 1)))
            else:
                self.list_of_clusters_to_follow[c] = np.vstack(
                    (self.list_of_clusters_to_follow[c], 
                     list_of_clusters_to_follow_temp[c]))
        return idx_temp_to_real

    def get_possible_positions(self, atom, c_feat):
        num_of_atom = c_feat[0]
        possibilities = np.array([], dtype=int)
        for i in range(num_of_atom):
            if (self.all_atom_features[atom] == \
                c_feat[1 + i * self.num_atom_features: \
                       1 + (i + 1) * self.num_atom_features]).all():
                possibilities = np.append(possibilities, i)
        return possibilities

    def search_clusters_to_follow(self, list_of_parents_visited, dict_of_bonds,
                                  father_idx, atom_number, atom_idx, features, 
                                  bond_matrix):
        # Make the list of all the new clusters to follow that are involving
        # the atom considered here.
        child = {}
        bonded_atoms = np.where(bond_matrix[atom_idx, :])[0]
        for i in dict_of_bonds[atom_number]:
            if i not in list_of_parents_visited:
                child[i] = []
                for ba in bonded_atoms:
                    if ba !=father_idx \
                        and (features[i] == self.all_atom_features[ba]).all():
                        child[i].append(ba)
                if child[i] == []:
                    return np.array([])
        if len(child) == 0:
            final_res = -np.ones([1, len(dict_of_bonds)], dtype=int)
            final_res[0, atom_number] = atom_idx
            return final_res
        res = {}
        size = []
        tot_size = 1
        key_to_order = {}
        order = 0
        for i in child.keys():
            key_to_order[order] = i
            order += 1
            res[i] = np.array([], dtype=int)
            res[i].shape = [0, len(dict_of_bonds)]
            for j in child[i]:
                subres = \
                    self.search_clusters_to_follow(\
                        np.append(list_of_parents_visited, atom_number), 
                        dict_of_bonds, atom_idx, i, j, features, bond_matrix)
                if subres.shape[0] != 0:
                    res[i] = np.vstack((res[i], subres))
            if res[i].shape[0] == 0:
                return np.array([])
            size.append(np.arange(res[i].shape[0]))
            tot_size *= res[i].shape[0]
        final_res = -np.ones([tot_size, len(dict_of_bonds)], dtype=int)
        final_res[:, atom_number] = atom_idx
        pos = -1
        for i in itertools.product(*size):
            pos += 1
            for j in range(len(i)):
                indexes = np.where(res[key_to_order[j]][i[j]] > -1)[0]
                final_res[pos, indexes] = res[key_to_order[j]][i[j], indexes]
        return final_res

    def add_to_list_of_reactants(self, bond_matrix, idx_temp_to_real):
        list_of_reactants_temp = {}
        
        reax_to_consider = self.get_reax_to_consider(idx_temp_to_real)
        
        for reax in reax_to_consider.keys():
            num_of_atoms_in_reax = self.reaction_dict[reax][0]
            for at_to_follow in reax_to_consider[reax]:
                all_combi, size_all_combi = self.get_combi(at_to_follow,
                                                           idx_temp_to_real,
                                                           reax)
                # Create a temporary list of reactants for this reaction,
                # only involving the reactants that would have one cluster
                # that just reacted.
                list_of_reactants_temp_2 = {}
                list_of_reactants_temp_2[reax] = \
                    self.create_list_of_reactants(all_combi, reax, 
                                                  size_all_combi, 
                                                  num_of_atoms_in_reax)
                if list_of_reactants_temp_2[reax].shape[0] != 0:
                    list_of_reactants_temp_2[reax] = \
                        self.correct_list_of_reactants(reax, 
                                                       num_of_atoms_in_reax, 
                                                       list_of_reactants_temp_2, 
                                                       bond_matrix)

                # Need this 'if' because 'correct_list_of_reactants' could have
                # removed all the reactants in list_of_reactants_temp_2.
                if list_of_reactants_temp_2[reax].shape[0] != 0:
                    if reax not in list_of_reactants_temp.keys():
                        list_of_reactants_temp[reax] = np.array([], dtype=int)
                        list_of_reactants_temp[reax].shape = \
                            [0, num_of_atoms_in_reax]
                    list_of_reactants_temp[reax] = np.vstack((\
                        list_of_reactants_temp[reax], 
                        list_of_reactants_temp_2[reax]))
        for reax in list_of_reactants_temp.keys():
            if list_of_reactants_temp[reax].shape[0] > 0:
                if len(list_of_reactants_temp[reax].shape) == 1:
                    list_of_reactants_temp[reax] = \
                        np.unique(list_of_reactants_temp[reax])
                else:
                    list_of_reactants_temp[reax] = \
                        self.rows_uniq_elems1(list_of_reactants_temp[reax])
                if list_of_reactants_temp[reax].shape[0] > 0:
                    list_of_reactants_temp[reax] = \
                        np.unique(list_of_reactants_temp[reax], axis=0)
                    if self.list_of_reactants[reax].shape[0] == 0:
                        if len(list_of_reactants_temp[reax].shape) == 1:
                            self.list_of_reactants[reax] = \
                                list_of_reactants_temp[reax].reshape(-1, 1)
                        else:
                            self.list_of_reactants[reax] = \
                                list_of_reactants_temp[reax]
                    elif self.list_of_reactants[reax].shape[1] == 1:
                        self.list_of_reactants[reax] = np.vstack(
                            (self.list_of_reactants[reax], 
                             list_of_reactants_temp[reax].reshape(-1, 1)))
                    else:
                        self.list_of_reactants[reax] = np.vstack(
                            (self.list_of_reactants[reax], 
                             list_of_reactants_temp[reax]))

    def get_reax_to_consider(self, idx_temp_to_real):
        # Give a dictionary containing the reactions that will need to be 
        # checked as one of the reactants was involved in the last reaction.
        # The dictionary has keys which are the indexes of the reactions to
        # check and the values are arrays containing the indexes of the clusters
        # of the reactions that need to be checked.
        reax_to_consider = {}
        for clusters_to_follow in idx_temp_to_real.keys():
            for reax in self.list_of_reactants.keys():
                for i in range(len(self.clusters_in_reactions[reax][0])):
                    if clusters_to_follow == \
                       self.clusters_in_reactions[reax][0][i]:
                        if reax in reax_to_consider.keys():
                            reax_to_consider[reax].append(i)
                        else:
                            reax_to_consider[reax] = [i]
        return reax_to_consider 


    def get_combi(self, at_to_follow, idx_temp_to_real, reax):
        # Get the combinations of all possible new reactions using the existing
        # clusters in the reaction and the ones that just appeared.
        all_combi = []
        size_all_combi = 1

        for i in range(len(self.clusters_in_reactions[reax][0])):
            if i == at_to_follow:
                all_combi.append(\
                    idx_temp_to_real[\
                        self.clusters_in_reactions[reax][0][i]])
                size_all_combi *= \
                    idx_temp_to_real[\
                        self.clusters_in_reactions[reax][0][i]].shape[0]
            else:
                all_combi.append(range(\
                    self.list_of_clusters_to_follow[\
                        self.clusters_in_reactions[reax][0][i]].shape[0]))
                size_all_combi *= \
                    self.list_of_clusters_to_follow[\
                        self.clusters_in_reactions[reax][0][i]].shape[0]
        return all_combi, size_all_combi

    def make_reaction(self, reaction_to_happen, h, t, bond_change, bond_matrix):
        # Update everything after a reaction.

        atoms_involved = self.pick_atoms_involved(reaction_to_happen)
        reax_descr = self.reaction_dict[reaction_to_happen]
        num_of_bond_change = (reax_descr.shape[0] - reax_descr[0] 
                              * self.atoms.num_atom_descriptors - 1) // 3

        for i in range(num_of_bond_change):
            reactant_1 = atoms_involved[reax_descr[ \
                reax_descr[0] * self.atoms.num_atom_descriptors + 1 + i * 3]]
            reactant_2 = atoms_involved[reax_descr[ \
                reax_descr[0] * self.atoms.num_atom_descriptors + 1 + i * 3 + 1]]
            change = reax_descr[ \
                reax_descr[0] * self.atoms.num_atom_descriptors + 1 + i * 3 + 2]
            if i == 0:
                bond_change[t] = np.array([[min(reactant_1, reactant_2), 
                                            max(reactant_1, reactant_2), 
                                            change]])
            else:
                bond_change[t] = np.vstack((bond_change[t], 
                     [min(reactant_1, reactant_2), max(reactant_1, reactant_2),
                     change]))

        h = self.update_all_after_reax(h,  bond_matrix, bond_change[t], t)
        return h, bond_change, bond_matrix

    def pick_atoms_involved(self, reaction_to_happen):
        idx_atoms_involved = np.random.randint( \
            self.list_of_reactants[reaction_to_happen].shape[0])
        atoms_involved \
            = self.list_of_reactants[reaction_to_happen][idx_atoms_involved, :]
        return atoms_involved

