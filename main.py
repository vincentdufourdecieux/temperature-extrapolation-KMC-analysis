import numpy as np

from KMC_simulation import run_KMC_simulation
from MD_analysis import run_analysis
from plot import plot_results


temperatures = [3400]

for j in range(len(temperatures)):
        # MD analysis info
        MD_analysis_bool = 0 # 1 if mechanism needs to be extracted from MD, 0 otherwise  
        atom_file = 'C4H10_896at_1_'+ str(temperatures[j]) + 'K_40GPa_50kts_0.12ts_every100_ffield.reax.cho.2017.atom' # Name of the MD file, must be in the '.atom' format and in the folder 'data/MD_data'
        folder_name_save = str(temperatures[j]) + 'K_1_6ps/' # Name of the folder in which analysis will be saved. It will be in the 'data/Results/(folder_name_save)' folder
        atom_features = np.array([1, 1]) # Boolean for features to describe atoms. The first is for atom type, the second is for the type of neighbors.
        reaction_features = np.array([1]) # Must be np.array([1])
        reaction_framework = 1 # Must be 1.
        start_MD_time = 0 # Time in ps for when the MD will be started to be memorized (molecules tracking)
        end_MD_time = 6 # End time in ps for when the MD will stop to be memorized
        start_analysis_time = 0 # Time in ps for when the MD mechanism will start to be extracted
        end_analysis_time = 6 # End time in ps for when the MD mechanism will stop to be extracted
        bond_duration_value = 8 # Number of steps for the bond duration
        bond_length_criteria = np.array([1.98, 1.57, 1.09]) # Length for the bond length criterion, in order length C-C bond, length C-H bond, length H-H bond
        timestep = 0.012 # Time in ps between two timesteps in the .atom file
        # molecules_to_track is 'all' if all molecules need to be tracked, otherwise string with molecules separated by ',' 
        # and each molecule is described by '(num of C)-(num of H)-(num of C-C)-(num of C-H)-(num of H-H)'.
        # For example, for only tracking CH4 and C2H6, write '1-4-0-4-0,2-6-1-6-0'
        molecules_to_track = 'all' 
        track_extra_variables = np.array([1, 1]) # Boolean for tracking other variables than molecules. The first is for the size of the longest molecule (in # of C) in th simulation, the second for the H:C ration in the longest molecule.
        max_atoms_in_reactions = 2 # The number of atoms in one reaction.

        # KMC run info
        KMC_bool = 1 # 1 if KMC needs to be run, 0 otherwise
        num_of_KMC = 1 # Number of KMC runs
        foldername_reactions = ['3600K_1_6ps/', '4000K_1_6ps/', '4500K_1_6ps/'] # Folder to use for the mechanism, if you want to do temperature extrapolation, you need to put several folders
        foldername_starting_molecules = str(temperatures[j]) + 'K_1_6ps/' # Folder to use for the system to study
        foldername_save_KMC = 'temp_extrap_to_3400K/KMC/' # Folder where to save the KMC
        molecules_to_track_KMC = 'all' # Same description as molecules_to_track
        track_extra_variables_KMC = np.array([1, 1]) # Same description as track_extra_variables
        reproduce_MD = 1 # If 1 the next variables of the KMC run info will be ignored and will be extracted from folder_starting_molecules
        end_t_sim = 6 # End time in ps of the KMC simulations
        # Following variables are only used in the case of temperature extrapolation (several folders in foldername_reactions)
        min_simulation_number_appearance = 3 # This defines how many times a reaction needs to appear in the different mechanisms to be considered.
        temp_values = np.array([3600, 4000, 4500]) # This defines the temperatures of the folders in foldername_reactions. MUST BE IN THE SAME ORDER
        temp_objective = 3400 # Temperature to which to extrapolate
        reactions_given = 0 # If 1, the path to a reaction dataframe must be given in reactions_dataframe to consider a set list of reactions (min_simulation_number_appearance is ignored then)
        reactions_dataframe = 'all' # Only considered if reactions_given is 1
        bootstrap_bool = 0 # 1 to run bootstrap KMC, 0 to run the baseline extrapolation
        num_of_test = 1000 # Number of bootstrap test.

        # Plot results
        plot_bool = 1 # 1 if the plots of molecules need to be plotted, 0 otherwise
        folder_MD = str(temperatures[j]) + 'K_1_6ps/MD_Results/' # Folder for the MD results that need to be plotted
        folder_KMC = foldername_save_KMC # Folder for the KMC results that need to be plotted
        num_of_KMC_to_plot = 1 # Number of KMC runs to plot
        molecules_to_plot ="4-10-3-10-0" # Which molecules to plot (same definition as in molecules_to_track but cannot be 'all')
        plot_extra_variables = np.array([1, 1]) # Boolean to plot the extra variables (same definition in track_extra_variables)

        if MD_analysis_bool == 1:
                run_analysis.main(atom_file, folder_name_save, atom_features, reaction_features, 
                                  reaction_framework, start_MD_time, end_MD_time, start_analysis_time, 
                                  end_analysis_time, bond_duration_value, bond_length_criteria, 
                                  timestep, molecules_to_track, track_extra_variables, max_atoms_in_reactions)

        if KMC_bool == 1:
                for i in range(1, num_of_KMC + 1):
                        a = 1
                        run_KMC_simulation.main(a, i, foldername_reactions, foldername_starting_molecules, 
                                                foldername_save_KMC, molecules_to_track_KMC, 
                                                track_extra_variables_KMC, reproduce_MD, end_t_sim, 
                                                min_simulation_number_appearance, temp_values, temp_objective, 
                                                reactions_given, reactions_dataframe, bootstrap_bool, num_of_test)

        if plot_bool == 1:
                plot_results.main(folder_MD, folder_KMC, num_of_KMC_to_plot, molecules_to_plot, 
                                  plot_extra_variables)



