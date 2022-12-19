import numpy as np
import time

from KMC_simulation import initialization_KMC, KMC_process
from util import save_data

def main(a, num, foldername_reactions, foldername_starting_molecules, 
        foldername_save, molecules_to_track_choice, 
        track_extra_variables, reproduce_MD, end_t_sim, 
        min_simulation_num_appearance, temp_values, temp_objective, 
        reactions_given, reactions_dataframe, bootstrap_bool, num_of_test):
    np.random.seed(a)
    t = time.process_time()

    if molecules_to_track_choice == 'all':
        molecules_to_track = np.array([])
    else:
        molecules_to_track = np.array([molecules_to_track_choice.split(',')])
    
    # Define the reactions, atoms, and molecules classes for the KMC. Load
    # other useful parameters.
    reactions_KMC, atoms_KMC, molecules_KMC, end_t, reaction_rates, \
    reaction_framework, first_frame \
        = initialization_KMC.get_all_for_KMC(foldername_reactions, 
            foldername_starting_molecules, foldername_save, molecules_to_track,
            track_extra_variables, min_simulation_num_appearance, temp_values,
            temp_objective, bootstrap_bool, num, num_of_test, 
            reproduce_MD, end_t_sim, reactions_given, reactions_dataframe)
    
    KMC_run = KMC_process.KMC(end_t, reactions_KMC, atoms_KMC, reaction_rates, 
                              first_frame)
    
    print("Starting KMC", flush=True)
    print(time.process_time(), flush=True)
    bond_change_KMC = KMC_run.run_KMC()
    print("End KMC", flush=True)
    print(time.process_time(), flush=True)

    time_range = molecules_KMC.time_range

    if molecules_to_track.shape[0] == 0:
        molecule_list = molecules_KMC.molecule_list
        molecule_per_frame = molecules_KMC.molecule_all_frames[ \
                            :time_range.shape[0], :molecule_list.shape[0]]
    else:
        molecule_list = molecules_KMC.molecules_to_track
        molecule_per_frame = molecules_KMC.molecule_all_frames[ \
                            :time_range.shape[0]]
    extra_variables = molecules_KMC.extra_variables[:time_range.shape[0]]

    save_data.save_data_KMC(foldername_save, bond_change_KMC, first_frame, 
                            molecule_per_frame, molecule_list, extra_variables,
                            time_range, num, a, foldername_reactions, 
                            foldername_starting_molecules, molecules_to_track,
                            track_extra_variables, 
                            min_simulation_num_appearance, temp_values, 
                            temp_objective, bootstrap_bool, num_of_test)


if __name__=='__main__':
    main()
