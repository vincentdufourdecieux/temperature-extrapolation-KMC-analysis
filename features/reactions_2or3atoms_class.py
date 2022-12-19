import itertools
import networkx as nx
import numpy as np
import pdb
import time

from util import constants


class Reactions:
    ''' Class defining the reactions descriptions. Here reactions can involve
    2 or 3 atoms. A reaction is described by:
    - the number of atoms involved
    - the descriptions of the atoms involved sorted by lexicographic order. So
    [1, 1, 3] will come before [2, 1, 0] which will come before [2, 2, 0].
    - the descriptions of the bonds that are evolving. This description will be
    the index of the first atom involved, the index of the second atom involved
    and 0 or 1 if the bond between the two atoms is broken or created, 
    respectively.
    
    For example, the reaction involving 3 atoms with descriptions [1, 3, 1], 
    [1, 1, 3], and [2, 1, 0], where the bond between [1, 1, 3] and [2, 1, 0] is
    broken and the bond between [1, 3, 1] and [2, 1, 0] is created will have 
    a description:
    [3, 1, 1, 3, 1, 3, 1, 2, 1, 0, 0, 2, 0, 1, 2, 1].
    '''
    def __init__(self, atom_features, reaction_features, num_of_types, \
                max_atoms_in_reactions, atoms, molecules, reaction_dict, \
                clusters_to_follow, clusters_in_reactions, \
                reaction_occurrences_per_frame, \
                atom_descriptions_in_reactions, \
                atom_descriptions_in_clusters_to_follow, 
                clusters_to_follow_per_atom_descriptions, MD_or_KMC):

        self.atom_features = atom_features
        self.reaction_features = reaction_features
        self.num_atom_descriptors = self.atom_features[0] \
                                    + num_of_types * self.atom_features[1]
        self.all_atom_descriptions = np.array([], dtype=int)

        # Set the maximum number of atoms that can involved in one reaction.
        self.max_atoms_in_reactions = max_atoms_in_reactions

        self.atoms = atoms
        self.molecules = molecules

        # Initialize (for MD analysis) or retrieve (for KMC simulation) the 
        # different descriptions of each reaction

        # Dictionary containing the description of the different reactions. 
        # The keys are the index of the reaction, the values are an array 
        # describing the reactions.
        self.reaction_dict = reaction_dict 

        # Number of times a reaction occur (size=[# of reactions])
        self.reaction_occurrences = np.array([], dtype=int)

        # Dictionary containing the frames at which a reaction occur.
        # The keys are the index of the reaction, the value are arrays
        # containing the frames at which a reaction occurred.
        self.reaction_occurrences_per_frame = reaction_occurrences_per_frame

        # Dictionary containing the description of a cluster of atoms that needs
        # to be followed as they appear in at least one reaction. These clusters
        # can contain between 1 and 3 atoms. Their description is:
        # - number of atoms in the cluster (1 to 3)
        # - atom descriptions of each of the atoms.
        # - bonds in the cluster: none for 1 atom, always 1 for 2 atoms, 
        # [binary if there is a bond between 1st and 2nd atom, 1st and 3rd atom,
        # 2nd and 3rd atom] for 3 atoms.
        # The keys are the index of the clusters to be followed,
        # the values are an array of their description. 
        self.clusters_to_follow = clusters_to_follow

        # Dictionary containing the clusters that are present in each reaction,
        # and the corresponding atoms in the reaction_dict description. This is
        # contained in a list of 2 elements. The first element is a list
        # containing the indexes of the clusters in the reaction. The second
        # element is a list that follows the order of the first element and 
        # gives in an array which atoms (in the order of reaction_dict) the
        # clusters contain. For example if the reaction 3 involves cluster 1 and
        # cluster 3 and that cluster 1 contains the first and second atom of
        # the description of reaction 3 (in reaction_dict) and that cluster 3
        # contains the third atom, then clusters_in_reactions[3] will be equal
        # to : [[1, 3], [array([0, 1]), array([2])].
        self.clusters_in_reactions = clusters_in_reactions

        # Dictionary containing the atom descriptions of the atoms involved in
        # the reaction. The keys are the reactions index and the values are a 
        # list of the descriptions of the atoms involved.
        self.atom_descriptions_in_reactions = atom_descriptions_in_reactions

        # Dictionary containing the atom descriptions of the atoms contained in
        # the clusters. The keys are the clusters index and the values are a
        # list of the descriptions of the atoms contained in the cluster.
        self.atom_descriptions_in_clusters_to_follow = \
            atom_descriptions_in_clusters_to_follow

        # Dictionary containing the clusters that contain a specific atom 
        # description. The keys are the atom descriptions and the values are
        # a list of the clusters indexes.
        self.clusters_to_follow_per_atom_descriptions = \
            clusters_to_follow_per_atom_descriptions

        self.list_of_clusters_to_follow = {}
        self.list_of_reactants = {}

        # 0 if MD analysis, 1 if KMC simulation
        self.MD_or_KMC = MD_or_KMC

    def update_reactions_and_real_bonds(self, bond_change_frame, real_bonds, frame):
        # Check the reactions that happened during the frame and get their description, if they happened before just
        # add an occurrence, otherwise add the reaction description to the list of reaction, their type and count an
        # occurrence.
        elementary_reactions_atoms, bond_changes_in_reactions \
                                = self.get_elementary_reactions(bond_change_frame)
        real_bonds_symmetric = (real_bonds + real_bonds.T).toarray()

        # For each elementary description, get its description, save it, if 
        # needed and update the different variables.
        for reax in range(len(elementary_reactions_atoms)):
            reax_description = self.get_reaction_description(reax, \
                                        elementary_reactions_atoms, \
                                        bond_changes_in_reactions, \
                                        real_bonds_symmetric)

            # If the reaction has strictly more than 3 atoms, go to the next 
            # one.
            if reax_description.shape[0] == 0:
                continue

            found = False
            for i in range(len(self.reaction_dict)):
                if np.array_equal(reax_description, self.reaction_dict[i]):
                    self.reaction_occurrences[i] += 1
                    self.reaction_occurrences_per_frame[i].append(frame)
                    found = True
                    break
            if not found:
                self.reaction_dict[len(self.reaction_dict)] = reax_description
                self.atom_descriptions_in_reactions[ \
                  len(self.reaction_dict) - 1] \
                    = np.unique( \
                        self.get_atom_descriptions_in_reaction( \
                            reax_description), axis=0)
                self.reaction_occurrences = np.append( \
                                                self.reaction_occurrences, 1)
                self.reaction_occurrences_per_frame[\
                                        len(self.reaction_dict) - 1] = [frame]
                self.update_clusters_to_follow_and_clusters_in_reactions( \
                                                            reax_description)
#        for i in range(bond_change_frame.shape[0]):
#            real_bonds[bond_change_frame[i, 0], bond_change_frame[i, 1]] = bond_change_frame[i, 2]

    def get_elementary_reactions(self, bond_change_frame):
        # Obtain atoms that are participating in a same elementary reaction 
        # (atoms involved in the same bond rearrangement) and the indexes of the
        # bond changes in bond_change_frame of these elementary reactions. For
        # example, if bond_change_frame = [[12, 14, 0], [1, 3, 1], [12, 15, 1]], 
        # then elementary_reactions_atoms = [[12, 14, 15], [1, 3]] and 
        # bond_changes_indexes_in_reactions = [[0, 2], [1]].
        elementary_reactions_atoms = []
        bond_changes_in_reactions = []

        G = nx.Graph()
        G.add_edges_from(bond_change_frame[:, :2])
        cc = nx.connected_components(G)
        for c in cc:
            elementary_reactions_atoms.append(list(c))
            reactions_with_atoms \
              = np.where(np.isin(bond_change_frame[:, :2], list(c)).any(-1))[0]
            bond_changes_in_reactions.append(
                        bond_change_frame[reactions_with_atoms])

        return elementary_reactions_atoms, bond_changes_in_reactions


    def get_reaction_description(self, reax, elementary_reactions_atoms, \
                                 bond_changes_in_reactions, real_bonds):
        # Get the description of the reaction depending on the features you are
        # interested in.
        atoms_in_reaction = elementary_reactions_atoms[reax]
        bond_changes_in_reaction = bond_changes_in_reactions[reax]
        num_bond_changes_reax = bond_changes_in_reaction.shape[0]

        num_atoms_reax = len(atoms_in_reaction)
        if num_atoms_reax > self.max_atoms_in_reactions:
            return np.array([])

        # Get the description of each atom in the reaction.
        atoms_descriptions = np.zeros([num_atoms_reax, \
                                       self.num_atom_descriptors], dtype=int)
        atom_idx = {}
        for num_atom, atom in enumerate(atoms_in_reaction):
            atoms_descriptions[num_atom] \
                            = self.atoms.get_atom_features(atom, real_bonds)
            atom_idx[atom] = num_atom

        # Get the order of the atoms descriptions, by sorting with the first
        # column, then by the second, ...
        order = np.lexsort( \
         atoms_descriptions[:, range(self.num_atom_descriptors - 1, -1, -1)].T)
        description, description_inverse, description_counts \
                        = np.unique(atoms_descriptions, return_inverse=True, \
                                    return_counts=True, axis=0)

        # If two or more atoms have the same description, there are several
        # possibilities for the reaction description, this case is treated
        # in different_descriptions_possible_case(). The other case is treated
        # in only_one_description_case().
        if (description_counts > 1).any() and num_atoms_reax > 2:
            reax_description \
                = self.different_descriptions_possible_case( \
                        description, description_inverse, atoms_descriptions, \
                        atom_idx, num_atoms_reax, num_bond_changes_reax,\
                        bond_changes_in_reaction, order)
        else:
            reax_description \
                = self.only_one_description_case(num_atoms_reax, \
                                                 num_bond_changes_reax, \
                                                 atoms_descriptions, \
                                                 bond_changes_in_reaction, \
                                                 atom_idx, order)
        return reax_description

    def different_descriptions_possible_case(self, description, \
                                             description_inverse, \
                                             atoms_descriptions, atom_idx, \
                                             num_atoms_reax, \
                                             num_bond_changes_reax, \
                                             bond_changes_in_reaction, order):
        # Obtain the reaction description if some atoms have the same
        # descriptions by running all the possible combinations of the reaction
        # description and keeping the smallest in lexicographic order.

        # Get all the possible descriptions
        array_poss = []
        for i in range(description.shape[0]):
            atoms_with_description = np.where(description_inverse == i)[0]
            atoms_with_description_order \
              = np.where((order[:, None] == atoms_with_description).any(-1))[0]
            possible_orders \
                        = itertools.permutations(atoms_with_description_order)
            array_poss.append(list(possible_orders))
        possibilities = list(itertools.product(*array_poss))

        # Get the reaction descriptions of all the possible configurations.
        size_reaction_description = 1 \
                                    + num_atoms_reax*self.num_atom_descriptors \
                                    + 3*num_bond_changes_reax
        reax_description_possibilities \
                = np.zeros([len(possibilities), size_reaction_description], \
                           dtype=int)

        for i in range(len(possibilities)):
            order_temp = order[np.concatenate(possibilities[i])]
            reax_description_possibilities[i, :] \
                = self.only_one_description_case(num_atoms_reax, \
                                                 num_bond_changes_reax, \
                                                 atoms_descriptions, \
                                                 bond_changes_in_reaction, \
                                                 atom_idx, order_temp)

        final_order = np.lexsort( \
                        reax_description_possibilities[:, \
                        range(size_reaction_description - 1, -1, -1) \
                        ].T)
        return reax_description_possibilities[final_order[0], :]

    def only_one_description_case(self, num_atoms_reax, num_bond_changes_reax, \
                                  atoms_descriptions, bond_changes_in_reaction,\
                                  atom_idx, order):
        # Obtain the reaction description when there is only one description
        # possible.
        size_reaction_description = 1 \
                                    + num_atoms_reax*self.num_atom_descriptors \
                                    + 3*num_bond_changes_reax
        reax_description = np.zeros([size_reaction_description], dtype=int)

        # First a reaction is described by the number of atoms involved in the
        # reaction.
        reax_description[0] = num_atoms_reax

        # Then it is described by the ordered atom descriptions.
        reax_description[1:1 + num_atoms_reax * self.num_atom_descriptors] \
                    = atoms_descriptions[order, :].reshape(-1)

        # Then it is described by the bonds that are modified.
        bond_changes_in_reaction_with_idx = bond_changes_in_reaction.copy()
        for i in range(bond_changes_in_reaction.shape[0]):
            for j in range(2):
                bond_changes_in_reaction_with_idx[i, j] \
                        = np.where(order \
                                   == atom_idx[ \
                                            bond_changes_in_reaction[i, j]])[0]
        bond_changes_in_reaction_with_idx[:, :2] \
                        = np.sort(bond_changes_in_reaction_with_idx[:, :2], \
                                  axis=1)
        order_reax = np.lexsort(bond_changes_in_reaction_with_idx[:, \
                        np.arange(
                        bond_changes_in_reaction_with_idx.shape[0] - 1, -1, -1)\
                        ].T)
        bond_changes_in_reaction_with_idx \
                        = bond_changes_in_reaction_with_idx[order_reax, 0:3]
        reax_description[1 + num_atoms_reax * self.num_atom_descriptors:] \
                                        = bond_changes_in_reaction_with_idx \
                                            .reshape(-1)
        return reax_description

    def get_atom_descriptions_in_reaction(self, reax_description):
        # Get the list of the atom descriptions for this reaction. 
        descriptions = np.zeros([reax_description[0], \
                                 self.num_atom_descriptors], dtype=int)
        for i in range(reax_description[0]):
            descriptions[i] = reax_description[1 \
                                    + self.num_atom_descriptors*i: 1 \
                                    + self.num_atom_descriptors*(i+1)]
        return descriptions

    def update_clusters_to_follow_and_clusters_in_reactions(self, reax_description):
        # Get the atom descriptions of the atoms in the reaction.
        atom_descriptions = self.get_atom_descriptions_in_reaction( \
                                                            reax_description)

        # Get the bond change part of the reaction description.
        bond_changes_reax = self.get_bond_changes_in_reaction(reax_description)

        # Get the list of the clusters of atoms that are involved in the
        # reaction before it occurs. 
        clusters_in_reactions, G \
            = self.get_clusters_in_reactions(bond_changes_reax)

        # Update clusters_to_follow, atom_descriptions_in_clusters_to_follow
        # and reaction_to_clusters_to_follow.
        reaction_to_clusters_to_follow = []
        for i in range(len(clusters_in_reactions)):
            # If the cluster has only one atom, check if it is already in 
            # clusters_to_follow. If yes, put the index of the 
            # clusters_to_follow in reaction_to_clusters_to_follow. Otherwise,
            # update clusters_to_follow, atom_descriptions_in_clusters_to_follow
            # and reaction_to_clusters_to_follow.
            if clusters_in_reactions[i].shape[0] == 1:
                reaction_to_clusters_to_follow \
                    = self.update_one_atom_in_cluster(
                        reaction_to_clusters_to_follow, atom_descriptions,
                        clusters_in_reactions[i])
            else:
                # Same if several atoms in cluster
                reaction_to_clusters_to_follow \
                    = self.update_several_atoms_in_cluster( \
                        reaction_to_clusters_to_follow, atom_descriptions, \
                        clusters_in_reactions[i], G)

        # Update clusters_in_reactions
        self.clusters_in_reactions[len(self.reaction_dict) - 1] \
                = [reaction_to_clusters_to_follow, clusters_in_reactions]

    def get_bond_changes_in_reaction(self, reax_description):
        num_atoms_reax = reax_description[0]
        num_atoms_descriptions_reax \
                            = 1 + num_atoms_reax * self.num_atom_descriptors
        bond_changes_reax = np.zeros([(reax_description.shape[0] \
                          - num_atoms_descriptions_reax)//3, 3], dtype=int)
        for i in range(bond_changes_reax.shape[0]):
            bond_changes_reax[i, :] = reax_description[ \
                num_atoms_descriptions_reax + 3 * i \
                : num_atoms_descriptions_reax + 3 * (i + 1)]
        return bond_changes_reax

    def get_clusters_in_reactions(self, bond_changes_reax):
        clusters_in_reactions = []
        bonded_atoms = bond_changes_reax[ \
                        np.where(bond_changes_reax[:, 2] == 0)[0], :2]
        G = nx.Graph()
        G.add_nodes_from(np.unique(bond_changes_reax[:, :2]))
        G.add_edges_from(bonded_atoms)
        connected_components = nx.connected_components(G)
        for cc in connected_components:
            clusters_in_reactions.append(np.array(list(cc)))
        return clusters_in_reactions, G

    def update_one_atom_in_cluster(self, reaction_to_clusters_to_follow, \
                                   atom_descriptions, clusters_in_reactions):
        found = False
        for j in self.clusters_to_follow.keys():
            if np.array_equal( \
                        atom_descriptions[clusters_in_reactions][0],
                        self.clusters_to_follow[j][1:]):
                found = True
                reaction_to_clusters_to_follow.append(j)
        if found == False:
            self.clusters_to_follow[len(self.clusters_to_follow)] \
                = np.hstack(([1], 
                        atom_descriptions[clusters_in_reactions][0]))
            self.atom_descriptions_in_clusters_to_follow[ \
                len(self.clusters_to_follow) - 1] = \
                    self.get_atom_descriptions_in_reaction( \
                        self.clusters_to_follow[ \
                            len(self.clusters_to_follow) - 1])
            reaction_to_clusters_to_follow.append( \
                        len(self.clusters_to_follow) - 1)

        return reaction_to_clusters_to_follow


    def update_several_atoms_in_cluster(self, reaction_to_clusters_to_follow, 
                                        atom_descriptions, \
                                        clusters_in_reactions, G):
        # Extract the cluster and featurize it in a 
        # consistent way when there are strictly more than 1 atom in
        # the cluster.
        num_atoms_cluster = clusters_in_reactions.shape[0]
        cluster = np.zeros([self.num_atom_descriptors \
                                       * num_atoms_cluster \
                                       + num_atoms_cluster \
                                       * (num_atoms_cluster - 1)//2 
                                       + 1], dtype=int)
        cluster[0] = num_atoms_cluster
        cluster[ \
            1:self.num_atom_descriptors * num_atoms_cluster + 1] = \
                atom_descriptions[np.sort(clusters_in_reactions), \
                    :].reshape(-1)
        for j in range(num_atoms_cluster - 1):
            for k in range(j + 1, num_atoms_cluster):
                if G.has_edge(clusters_in_reactions[j], 
                              clusters_in_reactions[k]):
                    idx = self.num_atom_descriptors * num_atoms_cluster\
                          + num_atoms_cluster * j - j * (j + 1)//2 \
                          + k - j 
                    cluster[idx] = 1

        # Check if the featurization of the cluster is in
        # self.clusters_to_follow, and update accordingly.
        found = False
        for j in self.clusters_to_follow.keys():
            if np.array_equal(cluster, self.clusters_to_follow[j]):
                found = True
                reaction_to_clusters_to_follow.append(j)
        if found == False:
            self.clusters_to_follow[len(self.clusters_to_follow)] \
                        = cluster
            self.atom_descriptions_in_clusters_to_follow[\
                len(self.clusters_to_follow) - 1] \
                    = self.get_atom_descriptions_in_reaction( \
                        self.clusters_to_follow[ \
                            len(self.clusters_to_follow) - 1])
            reaction_to_clusters_to_follow.append( \
                len(self.clusters_to_follow) - 1)
        return reaction_to_clusters_to_follow

    def get_h(self, first_frame, bond_change, start_frame, end_frame):
        # Get the number of times each reaction could have happened for each 
        # frame between start_frame and end_frame.

        # Initialize the h arrays.
        h_tot = np.zeros(len(self.reaction_dict), dtype=int)
        h_per_frame = np.zeros([len(self.reaction_dict), 
                                end_frame - start_frame], dtype=int)
        bond_matrix = first_frame.copy()

        # Get the descriptions of all the atoms
        self.all_atom_descriptions \
            = self.atoms.get_all_atom_features(bond_matrix)

        # For each cluster_to_follow, obtain all the combinations of atoms
        # that form the correct cluster_to_follow
        self.get_list_of_clusters_to_follow(bond_matrix)

        h_prev = self.get_h_frame(bond_matrix)
        h_per_frame[:, 0] = h_prev

        frame_counter = constants.get_h_percentage_counter

        for frame in range(start_frame, end_frame):

            if frame/(end_frame - start_frame) > frame_counter:
                print(frame, flush=True)
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
        # (adjacency matrix) and the descriptions of the atoms.
        for cluster in range(len(self.clusters_to_follow)):
            num_of_atom = self.clusters_to_follow[cluster][0]

            # Case where there is only one atom in the cluster
            if num_of_atom == 1:
                self.update_list_of_reactant_one_atom(cluster)

            # Case where there are several atoms in the cluster
            else:
                # Initialize the array that will store all the possible 
                # combinations of atoms that form one cluster with the good
                # description
                num_to_idx = self.get_num_to_idx(num_of_atom)
                possible_comb = np.array([], dtype=int)
                possible_comb.shape = [0, num_of_atom]
                visited = 0

                # Consider each possible bond in the cluster
                for i in range(len(num_to_idx)):
                    # Only the atoms that are bonded define the cluster, the
                    # other atoms can be bonded or not
                    if self.clusters_to_follow[cluster]\
                            [1 + num_of_atom*self.num_atom_descriptors + i] \
                                == 1:
                        # Get the index, in the order of the cluster, of the 
                        # atoms that are consider
                        idx_1 = num_to_idx[i][0]
                        idx_2 = num_to_idx[i][1]

                        # Get the description of the atoms
                        atom_1_description = self.clusters_to_follow[cluster]\
                            [1 + idx_1 * self.num_atom_descriptors: \
                                1 + (idx_1 + 1) * self.num_atom_descriptors]
                        atom_2_description = self.clusters_to_follow[cluster]\
                            [1 + idx_2 * self.num_atom_descriptors: \
                                1 + (idx_2 + 1) * self.num_atom_descriptors]

                        if visited == 0:
                            # If it is the first bond of the cluster, 
                            # initialize all the possible bonds between the 
                            # atoms of the correct descriptions.
                            possible_comb = self.initialize_possible_comb( \
                                possible_comb, atom_1_description, 
                                atom_2_description, bond_matrix, idx_1, idx_2, 
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
                                        idx_2, num_of_atom, atom_2_description)
                            elif possible_comb[0, idx_1] == -1 \
                                and possible_comb[0, idx_2] != -1:
                                possible_comb = \
                                    self.update_possible_comb_1atom_present(
                                        possible_comb, bond_matrix, idx_2, 
                                        idx_1, num_of_atom, atom_1_description)
                            else:
                                possible_comb \
                                    = self.update_possible_comb_noatom_present(\
                                        possible_comb, atom_1_description, 
                                        atom_2_description, bond_matrix, idx_1,
                                        idx_2, num_of_atom)
                    
                    # If two atoms of the cluster have the same description,
                    # the cases where the same two atoms are in one of the
                    # possible combinations must be removed.
                    if np.array_equal(atom_1_description, atom_2_description):
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
        # the correct description can be obtained from all_atom_descriptions.
        indexes = np.argwhere((self.all_atom_descriptions \
             == self.clusters_to_follow[cluster][ \
                1: 1 + self.num_atom_descriptors]).all(-1))
        if indexes.shape[0] == 0:
            self.list_of_clusters_to_follow[cluster] = np.array([])
        else:
            self.list_of_clusters_to_follow[cluster] \
                = np.reshape(np.concatenate(indexes), (-1, 1))

    def get_num_to_idx(self, num_of_atom):
        # In the cluster description, the order of the status of the bonds are
        # between atom 0 and 1, between atom 0 and 2,... between atom 1 and 2,
        # between atom 1 and 3...
        num_to_idx = []
        for i in range(num_of_atom - 1):
            for j in range(i + 1, num_of_atom):
                num_to_idx.append([i, j])
        return num_to_idx

    def initialize_possible_comb(self, possible_comb, atom_1_description, 
                                 atom_2_description, bond_matrix, idx_1, idx_2,
                                 num_of_atom):
        # To initialize all the possible combinations of the cluster, we start
        # by getting all the atoms that have the same description as the first
        # atom of the bond considered. Then, we check if the atoms they are 
        # bonded have the same description as the second atom of the bond
        # considered.
        indexes_atom_1_description \
            = np.argwhere((self.all_atom_descriptions == atom_1_description)\
                          .all(-1))
        for atom_1 in indexes_atom_1_description:
            bonded_atom_1 = np.where(bond_matrix[atom_1[0], :])[0]
            for ba in bonded_atom_1:
                if np.array_equal(atom_2_description, 
                                  self.all_atom_descriptions[ba]):
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
                                           atom_description_not_present):
        # If one of the atoms of the considered bond in the cluster has already
        # been searched, then we just need to check if the atoms that are in the
        # possible combinations are bonded to atoms that have the second
        # description than the bond.
        new_possible_comb = np.array([], dtype=int)
        new_possible_comb.shape = [0, num_of_atom]
        for j in range(possible_comb.shape[0]):
            bonded_atom_1 \
                = np.where(bond_matrix[possible_comb[j, idx_present], :])[0]
            for ba in bonded_atom_1:
                if np.array_equal(atom_description_not_present, 
                                  self.all_atom_descriptions[ba]):
                    comb = possible_comb[j, :]
                    comb[idx_not_present] = ba
                    new_possible_comb = np.vstack((new_possible_comb, comb))
        return new_possible_comb

    def update_possible_comb_noatom_present(self, possible_comb, 
                                            atom_1_description, 
                                            atom_2_description, bond_matrix, 
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
                                            atom_1_description,
                                            atom_2_description, bond_matrix, 
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
        # all_atom_descriptions, and list_of_clusters_to_follow are not 
        # initialize yet.
        if self.MD_or_KMC == 1:
            self.molecules.initialize_molecules(bond_matrix, 0)
        if self.all_atom_descriptions.shape[0] == 0:
            self.all_atom_descriptions \
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
                self.correct_list_of_reactants(reax, num_of_atoms_in_reax, \
                                               bond_matrix)
            h[reax] = self.list_of_reactants[reax].shape[0]

        return h

    def create_list_of_reactants(self, all_combi, reax, size_all_combi, num_of_atoms_in_reax):
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
                                  bond_matrix):
        # Remove the reactions that are not possible, counted several times...
        list_of_reactants_reax = self.list_of_reactants[reax].copy()
        num_of_bond_change = (self.reaction_dict[reax].shape[0] \
                              - num_of_atoms_in_reax \
                              * self.num_atom_descriptors - 1) // 3
        for i in range(num_of_bond_change):
            start_idx = 1 + num_of_atoms_in_reax * self.num_atom_descriptors \
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
        
        self.list_of_reactants[reax] = list_of_reactants_reax

    def update_bond_matrix(self, bond_matrix, bond_change):
        # Update the adjacency matrix after some bond changes.
        for reax in range(bond_change.shape[0]):
            bond_matrix[bond_change[reax][0], bond_change[reax][1]] = bond_change[reax][2]
            bond_matrix[bond_change[reax][1], bond_change[reax][0]] = bond_change[reax][2]

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
        self.atoms.update_all_atom_features(bond_matrix, bond_change, self.all_atom_descriptions)  # Update after so we count the reactants before the reactions happen
        idx_temp_to_real = self.add_to_list_of_clusters_to_follow(atoms_involved, bond_matrix)
        self.add_to_list_of_reactants(bond_matrix, idx_temp_to_real)
        for reax in range(len(self.list_of_reactants)):
            h[reax] = self.list_of_reactants[reax].shape[0]
        return h

    def remove_from_list_of_clusters_to_follow(self, atoms_involved):
        atoms_involved_in_clusters_to_follow = {}
        for atom in atoms_involved:
            atoms_involved_in_clusters_to_follow[atom] = []
            pdb.set_trace()
            for reax in \
                range(len(self.atom_descriptions_in_clusters_to_follow)):
                if ((self.all_atom_descriptions[atom] \
                    == self.atom_descriptions_in_clusters_to_follow[reax])\
                       .all(-1)).any(-1):
                    atoms_involved_in_clusters_to_follow[atom].append(reax)
        for atom in atoms_involved_in_clusters_to_follow.keys():
            for reax in atoms_involved_in_clusters_to_follow[atom]:
                self.list_of_clusters_to_follow[reax]  \
                    = np.delete(self.list_of_clusters_to_follow[reax], 
                                np.where((self.list_of_clusters_to_follow[reax]\
                                          == atom).any(-1))[0], axis=0)

    def remove_from_list_of_reactants(self, atoms_involved):
        atoms_involved_in_reactants = {}
        for atom in atoms_involved:
            atoms_involved_in_reactants[atom] = []
            for reax in range(len(self.atom_descriptions_in_reactions)):
                if ((self.all_atom_descriptions[atom] == self.atom_descriptions_in_reactions[reax]).all(-1)).any(-1):
                    atoms_involved_in_reactants[atom].append(reax)
        for atom in atoms_involved_in_reactants.keys():
            for reax in atoms_involved_in_reactants[atom]:
                self.list_of_reactants[reax] = np.delete(self.list_of_reactants[reax], np.where((self.list_of_reactants[reax] == atom).any(-1))[0], axis=0)


    def add_to_list_of_clusters_to_follow(self, atoms_involved, bond_matrix):
        atoms_involved_in_clusters_to_follow = {}
        list_of_clusters_to_follow_temp = {}
        idx_temp_to_real = {}
        for atom in atoms_involved:
            atoms_involved_in_clusters_to_follow[atom] = []
            for reax in range(len(self.atom_descriptions_in_clusters_to_follow)):
                if ((self.all_atom_descriptions[atom] == self.atom_descriptions_in_clusters_to_follow[reax]).all(-1)).any(-1):
                    atoms_involved_in_clusters_to_follow[atom].append(reax)
                    num_of_atom = self.clusters_to_follow[reax][0]
                    list_of_clusters_to_follow_temp[reax] = np.array([], dtype=int)
                    list_of_clusters_to_follow_temp[reax].shape = [0, num_of_atom]
        for atom in atoms_involved_in_clusters_to_follow.keys():
            for reax in atoms_involved_in_clusters_to_follow[atom]:
                num_of_atom = self.clusters_to_follow[reax][0]
                if num_of_atom == 1:
                    list_of_clusters_to_follow_temp[reax] = np.append(list_of_clusters_to_follow_temp[reax], atom)
                else:
                    dict_of_bonds = self.get_dict_of_bonds(num_of_atom, reax)
                    descriptions = np.vstack([ self.clusters_to_follow[reax][1+ self.num_atom_descriptors*i: 1 + self.num_atom_descriptors*(i+1)]for i in range(num_of_atom)])
                    atom_in_clusters_to_follow = self.get_possible_positions(atom, self.clusters_to_follow[reax])
                    for atom_position in atom_in_clusters_to_follow:
                        new_clusters_to_follow = self.search_clusters_to_follow(np.array([]), dict_of_bonds, -1, atom_position, atom, descriptions, bond_matrix)
                        if new_clusters_to_follow.shape[0] > 0:
                            list_of_clusters_to_follow_temp[reax] = np.vstack((list_of_clusters_to_follow_temp[reax], new_clusters_to_follow))
        for reax in list_of_clusters_to_follow_temp.keys():
            if list_of_clusters_to_follow_temp[reax].shape[0] > 0:
                if len(list_of_clusters_to_follow_temp[reax].shape) == 1:
                    list_of_clusters_to_follow_temp[reax] = np.unique(list_of_clusters_to_follow_temp[reax])
                else:
                    list_of_clusters_to_follow_temp[reax] = self.rows_uniq_elems1(list_of_clusters_to_follow_temp[reax])
                if list_of_clusters_to_follow_temp[reax].shape[0] > 0:
                    list_of_clusters_to_follow_temp[reax] = np.unique(list_of_clusters_to_follow_temp[reax], axis=0)
                    if reax not in idx_temp_to_real.keys():
                        idx_temp_to_real[reax] = np.array([], dtype=int)
                    idx_temp_to_real[reax]= np.append(idx_temp_to_real[reax], list(range(self.list_of_clusters_to_follow[reax].shape[0],
                                                                 self.list_of_clusters_to_follow[reax].shape[0]+ list_of_clusters_to_follow_temp[reax].shape[0])))
                    if self.list_of_clusters_to_follow[reax].shape[0] == 0:
                        if len(list_of_clusters_to_follow_temp[reax].shape) == 1:
                            self.list_of_clusters_to_follow[reax] = list_of_clusters_to_follow_temp[reax].reshape(-1, 1)
                        else:
                            self.list_of_clusters_to_follow[reax] = list_of_clusters_to_follow_temp[reax]
                    elif self.list_of_clusters_to_follow[reax].shape[1] == 1:
                        self.list_of_clusters_to_follow[reax] = np.vstack(
                            (self.list_of_clusters_to_follow[reax], list_of_clusters_to_follow_temp[reax].reshape(-1, 1)))
                    else:
                        self.list_of_clusters_to_follow[reax] = np.vstack((self.list_of_clusters_to_follow[reax], list_of_clusters_to_follow_temp[reax]))
        return idx_temp_to_real

    def get_possible_positions(self, atom, list):
        num_of_atom = list[0]
        possibilities = np.array([], dtype=int)
        for i in range(num_of_atom):
            if (self.all_atom_descriptions[atom] == list[1 + i*self.num_atom_descriptors: 1 + (i+1)*self.num_atom_descriptors]).all():
                possibilities = np.append(possibilities, i)
        return possibilities

    def get_dict_of_bonds(self, num_of_atom, reax):
        num_to_idx = self.get_num_to_idx(num_of_atom)
        connections = np.hstack(
            (num_to_idx, self.clusters_to_follow[reax][1 + num_of_atom * self.num_atom_descriptors:].reshape(-1, 1)))
        connections = np.delete(connections[:, :2], np.where(connections[:, 2] == 0)[0], axis=0)
        dict_of_bonds = {}
        for atom in range(num_of_atom):
            dict_of_bonds[atom] = []
            for bond in range(connections.shape[0]):
                if atom == connections[bond, 0]:
                    dict_of_bonds[atom].append(connections[bond,1])
                elif atom == connections[bond, 1]:
                    dict_of_bonds[atom].append(connections[bond, 0])
        return dict_of_bonds

    def search_clusters_to_follow(self, list_of_parents_visited, dict_of_bonds, father_idx, atom_number, atom_idx, descriptions, bond_matrix):
        child = {}
        bonded_atoms = np.where(bond_matrix[atom_idx, :])[0]
        for i in dict_of_bonds[atom_number]:
            if i not in list_of_parents_visited:
                child[i] = []
                for ba in bonded_atoms:
                    if ba !=father_idx and (descriptions[i] == self.all_atom_descriptions[ba]).all():
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
                subres = self.search_clusters_to_follow(np.append(list_of_parents_visited, atom_number), dict_of_bonds, atom_idx, i, j, descriptions, bond_matrix)
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
        reax_to_consider = {}
        list_of_reactants_temp = {}
        for clusters_to_follow in idx_temp_to_real.keys():
            for reax in self.list_of_reactants.keys():
                for i in range(len(self.clusters_in_reactions[reax][0])):
                    if clusters_to_follow == self.clusters_in_reactions[reax][0][i]:
                        if reax in reax_to_consider.keys():
                            reax_to_consider[reax].append(i)
                        else:
                            reax_to_consider[reax] = [i]
        for reax in reax_to_consider.keys():
            num_of_atoms_in_reax = self.reaction_dict[reax][0]
            for at_to_follow in reax_to_consider[reax]:
                all_combi = []
                size_all_combi = 1
                for i in range(len(self.clusters_in_reactions[reax][0])):
                    if i == at_to_follow:
                        all_combi.append(idx_temp_to_real[self.clusters_in_reactions[reax][0][i]])
                        size_all_combi *= idx_temp_to_real[self.clusters_in_reactions[reax][0][i]].shape[0]
                    else:
                        all_combi.append(range(self.list_of_clusters_to_follow[self.clusters_in_reactions[reax][0][i]].shape[0]))
                        size_all_combi *= self.list_of_clusters_to_follow[self.clusters_in_reactions[reax][0][i]].shape[0]
                list_of_reactants_temp_2 = self.create_list_of_reactants(all_combi, reax, size_all_combi, num_of_atoms_in_reax)
                if list_of_reactants_temp_2.shape[0] != 0:
                    list_of_reactants_temp_2 = self.correct_list_of_reactants(reax, num_of_atoms_in_reax, list_of_reactants_temp_2, bond_matrix)
                if list_of_reactants_temp_2.shape[0] != 0:
                    if reax not in list_of_reactants_temp.keys():
                        list_of_reactants_temp[reax] = np.array([], dtype=int)
                        list_of_reactants_temp[reax].shape = [0, num_of_atoms_in_reax]
                    list_of_reactants_temp[reax] = np.vstack((list_of_reactants_temp[reax], list_of_reactants_temp_2))
        for reax in list_of_reactants_temp.keys():
            if list_of_reactants_temp[reax].shape[0] > 0:
                if len(list_of_reactants_temp[reax].shape) == 1:
                    list_of_reactants_temp[reax] = np.unique(list_of_reactants_temp[reax])
                else:
                    list_of_reactants_temp[reax] = self.rows_uniq_elems1(list_of_reactants_temp[reax])
                if list_of_reactants_temp[reax].shape[0] > 0:
                    list_of_reactants_temp[reax] = np.unique(list_of_reactants_temp[reax], axis=0)
                    if self.list_of_reactants[reax].shape[0] == 0:
                        if len(list_of_reactants_temp[reax].shape) == 1:
                            self.list_of_reactants[reax] = list_of_reactants_temp[reax].reshape(-1, 1)
                        else:
                            self.list_of_reactants[reax] = list_of_reactants_temp[reax]
                    elif self.list_of_reactants[reax].shape[1] == 1:
                        self.list_of_reactants[reax] = np.vstack(
                            (self.list_of_reactants[reax], list_of_reactants_temp[reax].reshape(-1, 1)))
                    else:
                        self.list_of_reactants[reax] = np.vstack(
                            (self.list_of_reactants[reax], list_of_reactants_temp[reax]))



    def make_reaction(self, reaction_to_happen, h, t, bond_change, bond_matrix):
        atoms_involved = self.pick_atoms_involved(reaction_to_happen)
        num_of_bond_change = (self.reaction_dict[reaction_to_happen].shape[0] -
                              self.reaction_dict[reaction_to_happen][
                                  0] * self.atoms.num_atom_descriptors - 1) // 3
        for i in range(num_of_bond_change):
            reactant_1 = atoms_involved[self.reaction_dict[reaction_to_happen][
                self.reaction_dict[reaction_to_happen][0] * self.atoms.num_atom_descriptors + 1 + i * 3]]
            reactant_2 = atoms_involved[self.reaction_dict[reaction_to_happen][
                self.reaction_dict[reaction_to_happen][0] * self.atoms.num_atom_descriptors + 1 + i * 3 + 1]]
            change = self.reaction_dict[reaction_to_happen][
                self.reaction_dict[reaction_to_happen][0] * self.atoms.num_atom_descriptors + 1 + i * 3 + 2]
            if i == 0:
                bond_change[t] = np.array([[min(reactant_1, reactant_2), max(reactant_1, reactant_2), change]])
            else:
                bond_change[t] = np.vstack(
                    (bond_change[t], [min(reactant_1, reactant_2), max(reactant_1, reactant_2), change]))
        h = self.update_all_after_reax(h,  bond_matrix, bond_change[t], t)
        return [h, bond_change, bond_matrix]

    def pick_atoms_involved(self, reaction_to_happen):
        idx_atoms_involved = np.random.randint(self.list_of_reactants[reaction_to_happen].shape[0])
        atoms_involved = self.list_of_reactants[reaction_to_happen][idx_atoms_involved, :]
        return atoms_involved
