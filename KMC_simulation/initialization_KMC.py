import numpy as np
import pdb
import sys

from features import atom_class, molecule_class, reactions_compute_h_class
from KMC_simulation import temperature_extrapolation
from util import save_data


def get_all_for_KMC(foldername_reactions, foldername_starting_molecules, 
                    foldername_save, molecules_to_track, track_extra_variables,
                    min_simulation_number_appearance, temp_values, 
                    temp_objective, bootstrap_bool, num, num_of_test, 
                    reproduce_MD, end_t_sim, reactions_given,
                    reactions_dataframe):
    # Get parameters from the MD that is used to define the initial composition
    # and the parameters used for the analysis.
    first_frame, num_of_atoms, start_time_MD, end_time_MD, atoms_type, \
        timestep, volume = save_data.load_MD_parameters_starting_molecules(
                             foldername_starting_molecules)
    atom_features, reaction_features, num_of_types, reaction_framework, \
    max_atoms_in_reactions = save_data.load_analysis_parameters_reactions(\
                                foldername_reactions[0])

    start_frame_MD = int(start_time_MD/timestep)
    end_frame_MD = int(end_time_MD/timestep)

    # Initialize the atoms and the molecules.
    atoms_KMC = atom_class.Atoms(atom_features, num_of_types, atoms_type)
    molecules_KMC = molecule_class.Molecules(reaction_features, num_of_atoms, 
                                             num_of_types, atoms_type, 
                                             atoms_KMC, start_frame_MD, 
                                             end_frame_MD, molecules_to_track,
                                             track_extra_variables)

    # If there is only one folder in foldername_reactions, then there is no
    # extrapolation fit. If there are several then there is a 
    # temperature extrapolation fit.
    if reaction_framework == 1:
        if len(foldername_reactions) == 1:
            reaction_dict, clusters_to_follow, clusters_in_reactions, \
            atom_features_in_reactions, atom_features_in_clusters_to_follow, \
            reaction_rates \
                = save_data.load_analysis_data_1(foldername_reactions[0],
                                                 volume)
        else:
            reaction_dict, clusters_to_follow, clusters_in_reactions, \
                atom_features_in_reactions, atom_features_in_clusters_to_follow,\
                reaction_rates \
                    = temperature_extrapolation.get_reactions_from_several_MD(\
                        foldername_reactions, foldername_save, 
                        reaction_framework, timestep, temp_values, 
                        temp_objective, min_simulation_number_appearance, 
                        bootstrap_bool, num, num_of_test, volume, 
                        reactions_given, reactions_dataframe)

        reactions_KMC = reactions_compute_h_class.Reactions(
                            atom_features, reaction_features, num_of_types, 
                            max_atoms_in_reactions, atoms_KMC, molecules_KMC,
                            reaction_dict, clusters_to_follow, 
                            clusters_in_reactions, atom_features_in_reactions,
                            atom_features_in_clusters_to_follow, 1)

    # If we reproduce a given MD, the time of the KMC simulation is the same 
    # as the time of the MD. Otherwise, the time of the KMC simulation is given
    # by end_t_sim.
    if reproduce_MD == 1:
        end_t = end_time_MD - start_time_MD
    else:
        end_t = end_t_sim

    return reactions_KMC, atoms_KMC, molecules_KMC, end_t, reaction_rates, \
           reaction_framework, first_frame

