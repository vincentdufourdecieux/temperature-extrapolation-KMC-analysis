import numpy as np
import pdb
import time

from util import constants


class KMC:
    def __init__(self, end_t, reactions, atoms, reaction_rates, first_frame):
        self.end_t = end_t
        self.reactions = reactions
        self.atoms = atoms
        self.reaction_rates = reaction_rates
        self.first_frame = first_frame

    def run_KMC(self):
        # Run the all the frames of the KMC.
        h, propensity = self.initialize()
        t = 0
        bond_change_KMC = {}
        bond_matrix = self.first_frame.copy()
        time_counter = constants.time_KMC_percentage_counter
        while t < self.end_t:
            if t/self.end_t > time_counter:
                print(time_counter, flush=True)
                print(time.process_time(), flush=True)
                time_counter += constants.time_KMC_percentage_counter_increment
            reaction_to_happen, time_to_reaction \
                = self.pick_reaction_and_time(propensity)
            t += time_to_reaction
            propensity, h, bond_change_KMC, bond_matrix \
                = self.make_reaction(reaction_to_happen, h, t, bond_change_KMC,
                                     bond_matrix)
        return bond_change_KMC

    def initialize(self):
        # Initialize the KMC by getting the propensity at the first frame.
        bond_matrix = self.first_frame.copy()
        h = self.reactions.get_h_frame(bond_matrix)
        propensity = h * self.reaction_rates
        return h, propensity

    def pick_reaction_and_time(self, propensity):
        # Choose the reaction that is going to happen and the delay before it 
        # happens.
        propensity_tot = np.sum(propensity)
        r = np.random.random((2,))
        time_to_reaction = 1/propensity_tot*np.log(1/r[0])
        reaction_to_happen \
            = int(np.argwhere(np.cumsum(propensity) >= r[1]*propensity_tot)[0])
        return reaction_to_happen, time_to_reaction

    def make_reaction(self, reaction_to_happen, h, t, bond_change, bond_matrix):
        # Update propensity, bond_change_KMC and the bond_matrix
        h, bond_change, bond_matrix \
            = self.reactions.make_reaction(reaction_to_happen, h, t,
                                           bond_change, bond_matrix)
        propensity = h * self.reaction_rates
        return propensity, h, bond_change, bond_matrix

