import numpy as np
import os
import pandas as pd
import pdb
import pickle as pkl
from scipy import sparse

from util import constants


def save_data_analysis_framework_1(folder, start_time_MD, end_time_MD, 
                                   start_time_analysis, end_time_analysis,
                                   bond_length_criteria, bond_duration_value,
                                   num_of_frames, timestep, atom_file,
                                   atom_features, num_of_types, atoms_type,
                                   reaction_features, num_of_atoms,
                                   molecules_to_track, track_extra_variables,
                                   max_atoms_in_reactions, volume, 
                                   reaction_framework, bond_change,
                                   first_frame, first_frame_analysis,
                                   reaction_dict, reaction_rates,
                                   reaction_occurrences, h_tot,
                                   clusters_to_follow, 
                                   clusters_in_reactions,
                                   atom_features_in_reactions,
                                   atom_features_in_clusters_to_follow,
                                   molecules_per_frame, molecule_list,
                                   time_range, extra_variables):
    folder_analysis = create_folder_analysis(folder)
    folder_MD = create_folder_MD(folder)

    save_analysis_details(folder_analysis, start_time_analysis, 
                          end_time_analysis, 
                          bond_length_criteria, bond_duration_value,
                          num_of_frames, timestep, atom_file, atom_features,
                          num_of_types, atoms_type, reaction_features,
                          num_of_atoms, molecules_to_track,
                          track_extra_variables, max_atoms_in_reactions,
                          volume, reaction_framework)

    pkl.dump(bond_change, open(folder_MD + 'Bond_Change_MD.pkl', 'wb'))
    sparse.save_npz(folder_MD + 'First_Frame_MD.npz', 
                    sparse.csr_matrix(first_frame))
    sparse.save_npz(folder_analysis + 'First_Frame_Analysis.npz', 
            sparse.csr_matrix(first_frame_analysis))

    save_reactions(folder_analysis, reaction_dict, reaction_rates,
                   reaction_occurrences, h_tot)

    pkl.dump(clusters_to_follow, 
             open(folder_analysis + 'Clusters_To_Follow.pkl', 'wb'))
    pkl.dump(clusters_in_reactions, 
             open(folder_analysis + 'Clusters_In_Reactions.pkl', 'wb'))
    pkl.dump(atom_features_in_reactions,
             open(folder_analysis + 'Atom_Features_In_Reactions.pkl', 'wb'))
    pkl.dump(atom_features_in_clusters_to_follow,
             open(folder_analysis + 'Atom_Features_In_Clusters_To_Follow.pkl',
                  'wb'))

    sparse.save_npz(folder_MD + 'Molecules_Per_Frame_MD.npz',
                    sparse.csr_matrix(molecules_per_frame))
    np.save(folder_MD + 'Molecule_List_MD.npy', molecule_list)
    np.save(folder_MD + 'Time_Range_MD.npy', time_range)
    np.save(folder_MD + 'Extra_Variables_MD.npy', extra_variables)

    save_MD_details(folder_MD, start_time_MD, end_time_MD, num_of_frames, 
                    timestep, atom_file, num_of_types, atoms_type, 
                    num_of_atoms, volume, molecules_to_track,
                    track_extra_variables)

def create_folder_analysis(folder):
    res_path = constants.result_path
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    if folder[-1] != '/':
        folder = folder + '/'

    if not os.path.exists(res_path + folder):
        os.mkdir(res_path + folder)

    if not os.path.exists(res_path + folder + 'Analysis/'):
        os.mkdir(res_path + folder + 'Analysis/')

    return res_path + folder + 'Analysis/'

def save_analysis_details(folder, start_time_analysis, end_time_analysis, 
                          bond_length_criteria, bond_duration_value,
                          num_of_frames, timestep, atom_file, atom_features,
                          num_of_types, atoms_type, reaction_features,
                          num_of_atoms, molecules_to_track,
                          track_extra_variables, max_atoms_in_reactions,
                          volume, reaction_framework):
    analysis_details = {}
    analysis_details['Start_Time_Analysis'] = start_time_analysis
    analysis_details['End_Time_Analysis'] = end_time_analysis
    analysis_details['Bond_Length_Criteria'] = bond_length_criteria
    analysis_details['Bond_Duration_Value'] = bond_duration_value
    analysis_details['Num_Of_Frames'] = num_of_frames
    analysis_details['Timestep'] = timestep
    analysis_details['Atom_File'] = atom_file
    analysis_details['Atom_Features'] = atom_features
    analysis_details['Num_Of_Types'] = num_of_types
    analysis_details['Atoms_Type'] = atoms_type
    analysis_details['Reaction_Features'] = reaction_features
    analysis_details['Num_Of_Atoms'] = num_of_atoms
    analysis_details['Molecules_To_Track'] = molecules_to_track
    analysis_details['Track_Extra_Variables'] = track_extra_variables
    analysis_details['Max_Atoms_In_Reactions'] = max_atoms_in_reactions
    analysis_details['Volume'] = volume
    analysis_details['Reaction_Framework'] = reaction_framework

    pkl.dump(analysis_details, open(folder + 'Analysis_Details.pkl', 'wb'))

def save_reactions(folder, reaction_dict, reaction_rates, reaction_occurrences, 
                   h_tot):
    reaction_df = pd.DataFrame(0, index=np.arange(h_tot.shape[0]), 
                              columns=['Description', 'Rate', 'Occurrences', 
                                       'h'])
    for i in range(h_tot.shape[0]):
        descr = str(reaction_dict[i])
        descr = descr[1:-1]
        reaction_df.loc[i, 'Description'] = descr
        reaction_df.loc[i, 'Rate'] = reaction_rates[i]
        reaction_df.loc[i, 'Occurrences'] = reaction_occurrences[i]
        reaction_df.loc[i, 'h'] = h_tot[i]

    reaction_df.to_csv(folder + 'Reactions.csv')
        
def create_folder_MD(folder):
    res_path = constants.result_path

    if folder[-1] != '/':
        folder = folder + '/'

    if not os.path.exists(res_path + folder + 'MD_Results/'):
        os.mkdir(res_path + folder + 'MD_Results/')

    return res_path + folder + 'MD_Results/'

def save_MD_details(folder_MD, start_time_MD, end_time_MD, num_of_frames, 
                    timestep, atom_file, num_of_types, atoms_type, 
                    num_of_atoms, volume, molecules_to_track,
                    track_extra_variables):
    MD_details = {}
    MD_details['Start_Time_MD'] = start_time_MD
    MD_details['End_Time_MD'] = end_time_MD
    MD_details['Num_Of_Frames'] = num_of_frames
    MD_details['Timestep'] = timestep
    MD_details['Atom_File'] = atom_file
    MD_details['Num_Of_Types'] = num_of_types
    MD_details['Atoms_Type'] = atoms_type
    MD_details['Num_Of_Atoms'] = num_of_atoms
    MD_details['Volume'] = volume
    MD_details['Molecules_To_Track'] = molecules_to_track
    MD_details['Track_Extra_Variables'] = track_extra_variables

    pkl.dump(MD_details, open(folder_MD + 'MD_Details.pkl', 'wb'))

def load_MD_parameters_starting_molecules(folder):
    res_path = constants.result_path

    if folder[-1] != '/':
        folder = folder + '/'

    first_frame = sparse.load_npz(res_path + folder 
                                  + 'MD_Results/First_Frame_MD.npz').toarray()
    MD_details = pkl.load(open(res_path + folder + 'MD_Results/MD_Details.pkl',
                               'rb'))
    num_of_atoms = MD_details['Num_Of_Atoms']
    start_time_MD = MD_details['Start_Time_MD']
    end_time_MD = MD_details['End_Time_MD']
    atoms_type = MD_details['Atoms_Type']
    timestep = MD_details['Timestep']
    volume = MD_details['Volume']

    return first_frame, num_of_atoms, start_time_MD, end_time_MD, atoms_type,\
           timestep, volume 

def load_analysis_parameters_reactions(folder):
    res_path = constants.result_path

    if folder[-1] != '/':
        folder = folder + '/'

    analysis_details = pkl.load(open(res_path + folder 
                                     + 'Analysis/Analysis_Details.pkl', 'rb'))
    atom_features = analysis_details['Atom_Features']
    reaction_features = analysis_details['Reaction_Features']
    num_of_types = analysis_details['Num_Of_Types']
    reaction_framework = analysis_details['Reaction_Framework']
    max_atoms_in_reactions = analysis_details['Max_Atoms_In_Reactions']

    return atom_features, reaction_features, num_of_types, reaction_framework, \
           max_atoms_in_reactions

def load_analysis_data_1(folder, volume):
    res_path = constants.result_path

    if folder[-1] != '/':
        folder = folder + '/'

    df_reactions = pd.read_csv(res_path + folder + 'Analysis/Reactions.csv')
    reaction_dict = {}
    for r in range(df_reactions.shape[0]):
        reaction_dict[r] = np.array(df_reactions['Description'].iloc[r].split(),
                                    dtype=int) 

    clusters_to_follow = pkl.load(open(res_path + folder 
                                       + 'Analysis/Clusters_To_Follow.pkl',
                                       'rb'))

    clusters_in_reactions = pkl.load(open(res_path + folder 
                                       + 'Analysis/Clusters_In_Reactions.pkl',
                                       'rb'))

    atom_features_in_reactions = pkl.load(open(res_path + folder 
                        + 'Analysis/Atom_Features_In_Reactions.pkl', 'rb'))

    atom_features_in_clusters_to_follow = pkl.load(open(res_path + folder 
                        + 'Analysis/Atom_Features_In_Clusters_To_Follow.pkl',
                        'rb'))
    
    reaction_rates = np.zeros([df_reactions.shape[0]])
    for r in range(df_reactions.shape[0]):
        reaction_rates[r] = df_reactions['Rate'].iloc[r] \
                            / volume**(len(clusters_in_reactions[r][0]) - 1)

    return reaction_dict, clusters_to_follow, clusters_in_reactions, \
           atom_features_in_reactions, \
           atom_features_in_clusters_to_follow, reaction_rates

def save_data_KMC(folder, bond_change_KMC, first_frame, molecules_per_frame,
                  molecules_list, extra_variables, time_range, num, a, 
                  foldername_reactions, foldername_starting_molecules, 
                  molecules_to_track, track_extra_variables, 
                  min_simulation_num_appearance, x_values, x_objective, 
                  bootstrap_bool, num_of_test):
    folder_KMC = create_folder_KMC(folder)
   
    pkl.dump(bond_change_KMC,
            open(folder_KMC + 'Bond_Change_KMC_' + str(num) + '.pkl', 'wb'))
    sparse.save_npz(folder_KMC + 'First_Frame_KMC.npz', 
                    sparse.csr_matrix(first_frame))
    sparse.save_npz(folder_KMC + 'Molecules_Per_Frame_KMC_' + str(num) + '.npz',
                    sparse.csr_matrix(molecules_per_frame))
    np.save(folder_KMC + 'Molecule_List_KMC_' + str(num) + '.npy', 
            molecules_list)
    np.save(folder_KMC + 'Extra_Variables_KMC_' + str(num) + '.npy', 
            extra_variables)
    np.save(folder_KMC + 'Time_Range_KMC_' + str(num) + '.npy', time_range)

    save_KMC_details(folder_KMC, a, foldername_reactions, 
                     foldername_starting_molecules, molecules_to_track,
                     track_extra_variables, min_simulation_num_appearance,
                     x_values, x_objective, bootstrap_bool, num_of_test)

def create_folder_KMC(folder):
    res_path = constants.result_path
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    if folder[-1] != '/':
        folder = folder + '/'

    folder_cut = folder.split('/')
    folder_temp = ''
    for i in range(len(folder_cut)):
        folder_temp += folder_cut[i] + '/'
        if not os.path.exists(res_path + folder_temp):
            os.mkdir(res_path + folder_temp)

    return res_path + folder

def save_KMC_details(folder_KMC, a, foldername_reactions, 
                     foldername_starting_molecules, molecules_to_track,
                     track_extra_variables, min_simulation_num_appearance,
                     x_values, x_objective, bootstrap_bool, num_of_test):
    KMC_details = {}
    KMC_details['Random_Seed'] = a
    KMC_details['Foldername_Reactions'] = foldername_reactions
    KMC_details['Foldername_Starting_Molecules'] = foldername_starting_molecules
    KMC_details['Molecules_To_Track'] = molecules_to_track
    KMC_details['Track_Extra_Variables'] = track_extra_variables

    if len(foldername_reactions) > 1:
        KMC_details['Min_Simulation_Num_Appearance'] \
            = min_simulation_num_appearance
        KMC_details['X_Values'] = x_values
        KMC_details['X_Objective'] = x_objective
        KMC_details['Bootstrap_Bool'] = bootstrap_bool
        KMC_details['Num_Of_Test'] = num_of_test

    pkl.dump(KMC_details, open(folder_KMC + 'KMC_Details.pkl', 'wb'))


