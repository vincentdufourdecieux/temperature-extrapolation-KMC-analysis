import numpy as np
import os
import pandas as pd
import pickle as pkl
import pdb
import scipy.stats
import statsmodels.api as sm
import sys
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from util import constants, save_data

def get_reactions_from_several_MD(foldername_reactions, foldername_save, 
                                  reaction_framework, timestep, temp_values,
                                  temp_objective,
                                  min_simulation_number_appearance,
                                  bootstrap_bool, num, num_of_test,
                                  volume_starting_molecules,
                                  reactions_given, reactions_dataframe):
    # Obtain a data frame with the reactions that are common to at least 
    # min_simulation_number_appearance simulation analyses, and get the 
    # reaction rates, h, occurrences from all of them.
    if reactions_given == 0:
        df_reactions = get_df_reactions(foldername_reactions, 
                                        reaction_framework, 
                                        min_simulation_number_appearance,
                                        timestep)
    else:
        df_reactions = get_df_reactions_with_reactions_given(
                        foldername_reactions, timestep, reactions_dataframe)

   
    inv_temp_values = 1/np.array(temp_values)
    inv_temp_objective = 1/temp_objective
    if bootstrap_bool == 0:
        df_reactions = get_extrapolated_rates(df_reactions, inv_temp_values,
                                              inv_temp_objective)
    else:
        df_reactions = get_extrapolated_rates_with_bootstrap(df_reactions, 
                        inv_temp_values, inv_temp_objective, num_of_test,
                        foldername_reactions)
    
    if len(foldername_reactions) == min_simulation_number_appearance:
        clusters_to_follow, clusters_in_reactions,\
            atom_features_in_reactions, atom_features_in_clusters_to_follow \
                = get_important_dict([foldername_reactions[0]], 
                                           df_reactions)
    else:
        clusters_to_follow, clusters_in_reactions,\
            atom_features_in_reactions, atom_features_in_clusters_to_follow =\
                get_important_dict(foldername_reactions, df_reactions)

    # Save the list of reactions considered in  this KMC with the rates.
    folder = constants.result_path
    for subf in foldername_save.split('/'):
        folder += subf + '/'
        if not os.path.exists(folder):
            os.mkdir(folder)
    df_reactions.to_csv(folder + 'Reactions_' + str(num) + '.csv')

    # Volume correction for the rates and obtain reaction_dict.
    reaction_dict = {}
    reaction_rates = np.zeros([df_reactions.shape[0]])
    for r in range(df_reactions.shape[0]):
        reaction_rates[r] = df_reactions.loc[r, 'Rate'] \
                            / volume_starting_molecules \
                            ** (len(clusters_in_reactions[r][0]) - 1)
        reaction_dict[r] = np.array(df_reactions.loc[r, 'Description'].split(),
                                    dtype=int)

    return reaction_dict, clusters_to_follow, clusters_in_reactions, \
           atom_features_in_reactions, atom_features_in_clusters_to_follow,\
           reaction_rates

def get_df_reactions(foldername_reactions, reaction_framework, 
                     min_simulation_number_appearance, timestep):
    # Initialize the data frame.
    df_reactions = initialize_df_reactions(foldername_reactions)
    res_path = constants.result_path

    # Obtain the rates from each of the simulations and keep the reactions
    # that are common to at least min_simulation_number_appearance simulations.
    for i in range(len(foldername_reactions)):
        path_temp = res_path + foldername_reactions[i]
        f = foldername_reactions[i]
        
        # Load variables needed.
        details = pkl.load(open( \
            path_temp + 'Analysis/Analysis_Details.pkl', 'rb'))
        volume = details['Volume']
        df_temp = pd.read_csv(path_temp + 'Analysis/Reactions.csv')
        clusters_in_reactions = pkl.load(open( \
            path_temp + 'Analysis/Clusters_In_Reactions.pkl', 'rb'))

        # Check if reactions of df_temp are in df_reactions, if yes, add the
        # info from df_temp. If not, create a new row for the new reaction
        # and add the info from df_temp.
        for j in range(df_temp.shape[0]):
            found = False
            volume_term = volume**(len(clusters_in_reactions[i][0]) - 1)
            for k in range(df_reactions.shape[0]):
                if df_temp['Description'].iloc[j] \
                    == df_reactions['Description', ''].iloc[k]:
                    found = True
                    fill_column(df_reactions, k, f, df_temp, j, timestep, 
                                volume_term)
                    break
            if found == False:
                df_reactions.loc[df_reactions.shape[0]] = np.nan
                fill_column(df_reactions, df_reactions.shape[0] - 1, f, 
                            df_temp, j, timestep, volume_term)

    # Remove reactions that appear less than min_simulation_number_appearance
    # times.
    rates = df_reactions['Rate'].to_numpy(dtype=float)
    num_of_nans = np.count_nonzero(np.isnan(rates), axis = 1)
    to_drop = np.where(rates.shape[1] - num_of_nans \
                       < min_simulation_number_appearance)[0]
    df_reactions = df_reactions.drop(index=to_drop)
    df_reactions.index = np.arange(df_reactions.shape[0])

    return df_reactions

def initialize_df_reactions(foldername_reactions):
    # Initialize data frame. Each column will have a multi_index, the first
    # index is the folder of the simulation considered, the second one is 'Rate'
    # 'Occurrences', 'h', 'Error', and 'Error_log'. An additional index is meant
    # for the description of the reaction.
    multi_indexes = []
    for name in foldername_reactions:
        multi_indexes.append(['Rate', name])
        multi_indexes.append(['Occurrences', name])
        multi_indexes.append(['h', name])
        multi_indexes.append(['Error', name])
        multi_indexes.append(['Error_log', name])
    multi_indexes.append(['Description', ''])
    ind = pd.MultiIndex.from_tuples(multi_indexes)
    df_reactions = pd.DataFrame([], index=[], columns=ind)
    return df_reactions

def fill_column(df_reactions, k, f, df_temp, j, timestep, volume_term):
    # Fill the column of df_reactions with the corresponding column of 
    # df_temp.
    df_reactions.loc[k, ('Description', '')] = df_temp.loc[j, "Description"]
    df_reactions.loc[k, ('Rate', f)] = df_temp.loc[j, "Rate"]
    df_reactions.loc[k, ('Occurrences', f)] = df_temp.loc[j, ("Occurrences")]
    df_reactions.loc[k, ('h', f)] = df_temp.loc[j, "h"]
    df_reactions.loc[k, ('Error', f)] \
        = 1.96*np.sqrt(df_temp.loc[j, "Rate"]/df_temp.loc[j, "h"]/timestep)
    df_reactions.loc[k, ('Error_log', f)] \
        = - np.log10(np.exp(np.log(df_temp.loc[j, 'Rate']/volume_term) \
                            - 1.96/np.sqrt(df_temp.loc[j, "Occurrences"]))\
                            *volume_term) + np.log10(df_temp.loc[j, 'Rate'])

def get_df_reactions_with_reactions_given(foldername_reactions, timestep, 
                                          reactions_dataframe):
    df_reactions = initialize_df_reactions(foldername_reactions)
    res_path = constants.result_path
    df_reactions_given = pd.read_csv(res_path + reactions_dataframe)
    all_reax = df_reactions_given['Description']
    df_reactions['Description', ''] = df_reactions_given['Description']
    for i in range(len(foldername_reactions)):
        f = foldername_reactions[i]
        details = pkl.load(open(res_path + f 
                                + 'Analysis/Analysis_Details.pkl', 'rb'))
        volume = details['Volume']
        clusters_in_reactions \
            = pkl.load(open(res_path + f 
                            + 'Analysis/Clusters_In_Reactions.pkl', 'rb'))
        df_temp = pd.read_csv(res_path + f 
                              + 'Analysis/Reactions.csv')
        for j in range(df_temp.shape[0]):
            volume_term = volume**(len(clusters_in_reactions[i][0]) - 1)
            for k in range(df_reactions.shape[0]):
                if df_temp['Description'].iloc[j] \
                    == df_reactions['Description', ''].iloc[k]:
                    fill_column(df_reactions, k, f, df_temp, j, timestep, 
                                volume_term)
                    break
    return df_reactions

def get_important_dict(foldername_reactions, df_reactions):
    # Get all the important dictionaries that are important and adapt them to
    # the newly extracted df_reactions

    # Initialize the new important dictionaries.
    clusters_to_follow = {}
    clusters_in_reactions = {}
    atom_features_in_reactions = {}
    atom_features_in_clusters_to_follow = {}
    idx_clusters_to_follow = {}
    inv_clusters_to_follow = {}
    count_clusters_to_follow = 0
    res_folder = constants.result_path

    for subf in foldername_reactions:
        # For each folder that is part of the extrapolation, load the 
        # important dictionaries.
        f = res_folder + subf
        df_reactions_old = pd.read_csv(f + 'Analysis/Reactions.csv')
        clusters_to_follow_old \
            = pkl.load(open(f + 'Analysis/Clusters_To_Follow.pkl', 'rb'))
        clusters_in_reactions_old \
            = pkl.load(open(f + 'Analysis/Clusters_In_Reactions.pkl', 'rb'))
        atom_features_in_reactions_old \
            = pkl.load(open(f \
                + 'Analysis/Atom_Features_In_Reactions.pkl', 'rb'))
        atom_features_in_clusters_to_follow_old \
            = pkl.load(open(f \
                + 'Analysis/Atom_Features_In_Clusters_To_Follow.pkl', 'rb'))

        # For each reaction in the new df_reactions, check if it was in the
        # folder considered and store the corresponding idx.
        idx_new = - np.ones([df_reactions.shape[0]], dtype=int)
        inv_new = - np.ones([df_reactions_old.shape[0]], dtype=int)
        for i in range(df_reactions_old.shape[0]):
            reax_old = df_reactions_old.loc[i, 'Description']
            for j in range(df_reactions.shape[0]):
                reax = df_reactions.loc[j, 'Description']
                if reax == reax_old:
                    idx_new[j] = i
                    inv_new[i] = j


        for i in range(df_reactions.shape[0]):
            if idx_new[i] != -1 and i not in atom_features_in_reactions.keys():
                atom_features_in_reactions[i] \
                    = atom_features_in_reactions_old[idx_new[i]]
                clusters_in_reactions[i] \
                    = clusters_in_reactions_old[idx_new[i]]
                for num_j, j in \
                    enumerate(clusters_in_reactions_old[idx_new[i]][0]):
                    if str(clusters_to_follow_old[j]) \
                        not in inv_clusters_to_follow.keys():
                        inv_clusters_to_follow[str(clusters_to_follow_old[j])] \
                            = count_clusters_to_follow
                        clusters_to_follow[count_clusters_to_follow] \
                            = clusters_to_follow_old[j]
                        atom_features_in_clusters_to_follow[\
                            count_clusters_to_follow] \
                                = atom_features_in_clusters_to_follow_old[j]
                        count_clusters_to_follow += 1
                    clusters_in_reactions[i][0][num_j] \
                        = inv_clusters_to_follow[str(clusters_to_follow_old[j])]
    return clusters_to_follow, clusters_in_reactions, \
           atom_features_in_reactions, atom_features_in_clusters_to_follow

def get_extrapolated_rates(df_reactions, inv_temp_values, inv_temp_objectives):
    df_reactions_final = pd.DataFrame(-1, 
                                      index=np.arange(df_reactions.shape[0]),
                                      columns=['Description', 'Rate',
                                               'Occurrences', 'h'])

    df_reactions_final['Description'] = df_reactions[('Description', '')]
    df_reactions_final['Occurrences'] = np.sum(df_reactions['Occurrences'], 
                                               axis=1)
    df_reactions_final['h'] = np.sum(df_reactions['h'], axis=1)

    for i in range(df_reactions.shape[0]):
        x = sm.add_constant(inv_temp_values)
        x_obj = np.array([1, inv_temp_objectives])
        y = np.log10(np.array(df_reactions['Rate'].iloc[i], dtype=float))
        if np.where(np.isnan(y))[0].shape[0] > 0:
            x = np.delete(x, np.where(np.isnan(y))[0], axis=0)
            y = np.delete(y, np.where(np.isnan(y))[0], axis=0)

        #  Calculate model fit:
        try:
            lm_fit = sm.OLS(y, x).fit()
        except:
            pdb.set_trace()
        dt = lm_fit.get_prediction(x_obj).summary_frame()
        y_prd = dt['mean'][0]
        df_reactions_final.loc[i, 'Rate'] = 10**y_prd

        # Arrhenius or zero-barrier model
        reax = np.array(df_reactions_final['Description'].iloc[i].split(),
                        dtype=int)
        model_reaction = get_model_reaction(reax)
        if model_reaction == 1:
            df_reactions_final.loc[i, 'Rate'] = 10 ** np.mean(y)
    return df_reactions_final

def get_extrapolated_rates_with_bootstrap(df_reactions, inv_temp_values, 
                                          inv_temp_objective, num_of_test,
                                          foldername_reactions):
    df_reactions_final = pd.DataFrame(-1, 
                                      index=np.arange(df_reactions.shape[0]),
                                      columns=['Description', 'Rate',
                                               'Occurrences', 'h'])
    df_reactions_final['Description'] = df_reactions[('Description', '')]
    df_reactions_final['Occurrences'] = np.sum(df_reactions['Occurrences'],
                                               axis=1)
    df_reactions_final['h'] = np.sum(df_reactions['h'], 
                                     axis=1)
    for reax in range(df_reactions.shape[0]):

        points_temp = np.zeros([inv_temp_values.shape[0], num_of_test])
        results_bootstrap = np.zeros([num_of_test, 2])
        temp_values_temp = inv_temp_values

        suppressed = 0
        to_delete = []
        for j, f in enumerate(foldername_reactions):
            # No sampling of the folders that don't have a rate for this
            # reaction.
            if np.isnan(df_reactions['Rate', f].iloc[reax]):
                points_temp = np.delete(points_temp, -1, axis=0)
                to_delete.append(j)
                suppressed += 1
                continue

            # Sample num_of_test times in a gaussian centered on the log of 
            # the rate and with standard deviation the Error_log.
            points_temp[j - suppressed] \
                = scipy.stats.norm.rvs(size=num_of_test, 
                    loc=np.log10(df_reactions['Rate', f].iloc[reax]), 
                    scale=df_reactions['Error_log',f].iloc[reax] / 1.96)

        temp_values_temp = np.delete(temp_values_temp, to_delete)
        reaction_descr = np.array(df_reactions['Description'].iloc[reax].split(),
                                  dtype=int)
        model_reaction = get_model_reaction(reaction_descr)
        
        # For each of the test, extrapolate the reaction rate in temperature
        # using an Arrhenius model (model_reaction = 0) or a zero-barrier
        # model (model_reaction = 1). 
        for i in range(num_of_test):
            x = sm.add_constant(temp_values_temp)
            y = points_temp[:, i]
            if model_reaction == 0:
                x_goal = np.array([1, inv_temp_objective])
                lm_fit = sm.OLS(y, x).fit()
                dt = lm_fit.get_prediction(x_goal).summary_frame(alpha=0.05)
                results_bootstrap[i, 0] = np.array(dt['mean'][0])
                results_bootstrap[i, 1] = np.array(dt['mean_se'][0])
            elif model_reaction == 1:
                results_bootstrap[i, 0] = np.mean(y)
                results_bootstrap[i, 1] = scipy.stats.sem(y)

        # Use all the extrapolation to obtain the mean extrapolation at 
        # the desired temperature and the confidence interval standard deviation
        # to then sample a rate in the Gaussian centered on the mean
        # extrapolation and with the CI std as the std.
        results = np.zeros([2])
        results[0] = np.mean(results_bootstrap[:, 0])
        results[1] = np.mean(results_bootstrap[:, 1])
        if model_reaction == 0:
            df_reactions_final.loc[reax, 'Rate'] = 10 ** (
                scipy.stats.t.rvs(temp_values_temp.shape[0] - 2, 
                                  loc=results[0], size=1, scale=results[1]))
        elif model_reaction == 1:
            df_reactions_final.loc[reax, 'Rate'] = 10 ** (
                scipy.stats.t.rvs(temp_values_temp.shape[0] - 1,
                                  loc=results[0], size=1, scale=results[1]))
    return df_reactions_final

def get_model_reaction(reax):
    # If reaction_model == 1, then the temperature extrapolation is a zero
    # barrier model. If it is 0 then the temperature extrapolation is Arrhenius.
    if reax[0] == 2:
        undercoord_atoms = 0
        overcoord_atoms = 0
        overcoord_H = 0
        overcoord_C_less_than_1C = 0
        if reax[9] == 1:
            if reax[1] == 1:
                if reax[2] + reax[3] < 4:
                    undercoord_atoms += 1
            elif reax[1] == 2:
                if reax[2] + reax[3] < 1:
                    undercoord_atoms += 1
            if reax[4] == 1:
                if reax[5] + reax[6] < 4:
                    undercoord_atoms += 1
            elif reax[4] == 2:
                if reax[5] + reax[6] < 1:
                    undercoord_atoms += 1
        elif reax[9] == 0:
            if reax[1] == 1:
                if reax[2] + reax[3] > 4:
                    overcoord_atoms += 1
                    if reax[2] < 2:
                        overcoord_C_less_than_1C = 1
            elif reax[1] == 2:
                if reax[2] + reax[3] > 1:
                    overcoord_atoms += 1
                    overcoord_H = 1
            if reax[4] == 1:
                if reax[5] + reax[6] > 4:
                    overcoord_atoms += 1
                    if reax[5] < 2:
                        overcoord_C_less_than_1C = 1
            if reax[4] == 2:
                if reax[5] + reax[6] > 1:
                    overcoord_atoms += 1
                    overcoord_H = 1
        if undercoord_atoms == 2 or overcoord_atoms == 2 or overcoord_H == 1 or overcoord_C_less_than_1C == 1:
            return 1
    return 0
