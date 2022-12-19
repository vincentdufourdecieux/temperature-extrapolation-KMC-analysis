# Standard library imports
import numpy as np
import time

# Local application imports
from MD_analysis.initialization_MD_analysis import initialize_system
from features import atom_class, molecule_class, reactions_create_class, \
                     reactions_compute_h_class
from MD_analysis import bond_analysis_class
from util import save_data

def main(atom_file, folder_save, atom_features, reaction_features, reaction_framework, 
         start_MD_time, end_MD_time, start_analysis_time, end_analysis_time, 
         bond_duration_value, bond_length_criteria, timestep, molecules_to_track_choice, 
         track_extra_variables, max_atoms_in_reactions):

    if molecules_to_track_choice == 'all':
        molecules_to_track = np.array([])
    else:
        molecules_to_track = np.array([molecules_to_track_choice.split(',')])
    
    # Initialize system and get MD data
    pipeline, num_of_atoms, num_of_frames, num_of_types, atoms_type = \
                        initialize_system(atom_file, bond_length_criteria)

    # Convert times to frame numbers.
    start_frame_MD, end_frame_MD, start_frame_analysis, end_frame_analysis = \
                convert_times_to_frame_numbers(start_MD_time, end_MD_time, 
                                       start_analysis_time, end_analysis_time,
                                       num_of_frames, timestep)

    # Initialize atom description
    atoms = atom_class.Atoms(atom_features, num_of_types, atoms_type)

    # Initialize molecule description
    molecules = molecule_class.Molecules(reaction_features, num_of_atoms, \
                                         num_of_types, atoms_type, atoms, \
                                         start_frame_MD, end_frame_MD, \
                                         molecules_to_track, \
                                         track_extra_variables)

    # Initialize reaction description
    if reaction_framework == 1:
        reactions = reactions_create_class.Reactions(atom_features, \
                                                        reaction_features, \
                                                        num_of_types, \
                                                        max_atoms_in_reactions,\
                                                        atoms)
    print("Beginning bond analysis", flush=True)
    print(time.process_time(), flush=True)
    bond_analysis = bond_analysis_class.BondAnalysis(pipeline, 
                                               bond_duration_value, \
                                               num_of_atoms, reactions, \
                                               molecules, start_frame_MD, \
                                               end_frame_MD, \
                                               start_frame_analysis, \
                                               end_frame_analysis)
    first_frame, bond_change, first_frame_analysis, volume = \
                        bond_analysis.run_bond_analysis_and_extract_reactions()

    print("End bond analysis", flush=True)
    print(time.process_time(), flush=True)

    # Make first_frame arrays symmetric
    first_frame = (first_frame.T + first_frame).toarray()
    first_frame_analysis \
        = (first_frame_analysis.T + first_frame_analysis).toarray()

    # Go through all the time steps to know the number of times each reaction
    # could have happened.
    print("Beginning get h", flush=True)
    print(time.process_time(), flush=True)
    if reaction_framework == 1:
        reactions_to_compute = reactions_compute_h_class.Reactions( \
            atom_features, reaction_features, num_of_types, 
            max_atoms_in_reactions, atoms, molecules, reactions.reaction_dict,
            reactions.clusters_to_follow, reactions.clusters_in_reactions,
            reactions.atom_features_in_reactions,
            reactions.atom_features_in_clusters_to_follow, 0)
        h_tot, h_per_frame = reactions_to_compute.get_h(first_frame_analysis, 
                                                        bond_change,
                                                        start_frame_analysis, 
                                                        end_frame_analysis)
    print("End get h", flush=True)
    print(time.process_time(), flush=True)

    # Calculate reaction rates
    reaction_rates = np.zeros([h_tot.shape[0]])
    for r in range(h_tot.shape[0]):
        reaction_rates[r] = reactions.reaction_occurrences[r] \
                            * volume \
                            **(len(reactions.clusters_in_reactions[r][0])-1)\
                            / (h_tot[r] * timestep)

    # Save data
    time_range = molecules.time_range

    if molecules_to_track.shape[0] == 0:
        molecule_list = molecules.molecule_list
        molecule_all_frames = molecules.molecule_all_frames[ \
                                :time_range.shape[0], :molecule_list.shape[0]]
    else:
        molecule_list = molecules.molecules_to_track
        molecule_all_frames = molecules.molecule_all_frames[ \
                                :time_range.shape[0]]
    extra_variables = molecules.extra_variables[:time_range.shape[0]]

    if reaction_framework == 1:
        save_data.save_data_analysis_framework_1(
            folder_save, start_MD_time, end_MD_time, start_analysis_time, 
            end_analysis_time, bond_length_criteria, bond_duration_value,
            num_of_frames, timestep, atom_file, atom_features, num_of_types, 
            atoms_type, reaction_features, num_of_atoms, molecules_to_track,
            track_extra_variables, max_atoms_in_reactions, volume, 
            reaction_framework, bond_change,
            first_frame, first_frame_analysis, reactions.reaction_dict,
            reaction_rates, reactions.reaction_occurrences, h_tot,
            reactions.clusters_to_follow, reactions.clusters_in_reactions,
            reactions.atom_features_in_reactions,
            reactions.atom_features_in_clusters_to_follow,
            molecule_all_frames, molecule_list,
            time_range, extra_variables)

def convert_times_to_frame_numbers(start_time_MD, end_time_MD, 
                                   start_time_analysis, end_time_analysis,
                                   num_of_frames, timestep):
    # Convert start and end times to frame numbers. Check that these numbers 
    # are realistic.
                           
    assert start_time_MD <= end_time_MD, "Start time bigger than end time"
    assert start_time_analysis <= end_time_analysis, \
                                         "Start time bigger than end time"

    # Convert time to frame number
    start_frame_MD = int(start_time_MD/timestep)
    end_frame_MD = int(end_time_MD/timestep)
    start_frame_analysis = int(start_time_analysis/timestep)
    end_frame_analysis = int(end_time_analysis/timestep)
    
    # Check that the start frames are less than the total number of frames.
    # Correct the end frames if they are more than the number of frames in the
    # MD simulation.
    assert start_frame_MD < num_of_frames, \
                                    "Start frame bigger than number of frames"
    assert start_frame_analysis < num_of_frames, \
                                    "Start frame bigger than number of frames"
    if end_frame_analysis > num_of_frames:
        end_frame_analysis = num_of_frames
    if end_frame_MD > num_of_frames:
         end_frame_MD = num_of_frames
    
    return start_frame_MD, end_frame_MD, start_frame_analysis, \
           end_frame_analysis

if __name__=='__main__':
    main()
