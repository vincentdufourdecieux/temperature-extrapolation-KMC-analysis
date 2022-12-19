# Standard library imports
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pdb
import pickle as pkl
from scipy import sparse
import sys

# Local application imports
from util import constants

def main(folder_MD, folder_KMC, num_of_KMC, molecules_to_plot_str, 
         plot_extra_variables):

    molecules_to_plot = np.array(molecules_to_plot_str.split(','))
    num_molecules_to_plot = molecules_to_plot.shape[0]
    num_extra_variables_to_plot = np.sum(plot_extra_variables)
    num_to_plot = num_molecules_to_plot + num_extra_variables_to_plot

    mol_to_plot_MD, time_range_MD = get_data(folder_MD, molecules_to_plot,
                                                plot_extra_variables, 
                                                num_molecules_to_plot,
                                                num_extra_variables_to_plot,
                                                num_to_plot, 'MD', 0)

    mol_to_plot_KMC = np.zeros([num_of_KMC, time_range_MD.shape[0],num_to_plot])
    for i in range(num_of_KMC):
        mol_to_plot_KMC_temp, time_range_KMC_temp = get_data(
                folder_KMC, molecules_to_plot, plot_extra_variables, 
                num_molecules_to_plot, num_extra_variables_to_plot, num_to_plot,
                'KMC', i + 1)
        for j in range(num_to_plot):
            mol_to_plot_KMC[i, :, j] = np.interp(time_range_MD,
                                                 time_range_KMC_temp,
                                                 mol_to_plot_KMC_temp[:, j])

    mol_to_plot_KMC = np.mean(mol_to_plot_KMC, axis=0)

    for i in range(num_to_plot):
        fig = go.Figure() 
        fig.add_trace(go.Scatter(x=time_range_MD, y=mol_to_plot_MD[:, i],
                                mode='lines', name='MD'))
        fig.add_trace(go.Scatter(x=time_range_MD, y=mol_to_plot_KMC[:, i],
                                mode='lines', name='KMC'))
        name_molecule = ''
        if i < num_molecules_to_plot:
            molecule_count = np.array(molecules_to_plot[i].split('-'), dtype=int)
            for j in range(2):
                if molecule_count[j] != 0:
                    if j == 0:
                        name_molecule += 'C' + str(molecule_count[j])
                    elif j == 1:
                        name_molecule += 'H' + str(molecule_count[j])
        if num_extra_variables_to_plot !=0:
            if i == num_molecules_to_plot and plot_extra_variables[0] == 1:
                name_molecule = 'Num of C in longest molecule'
            elif i == num_molecules_to_plot and plot_extra_variables[1] == 1:
                name_molecule = 'Ratio H/C in longest molecule'
            elif i == num_molecules_to_plot + 1:
                name_molecule = 'Ratio H/C in longest molecule'
        fig.update_layout(
            title=name_molecule,
            xaxis_title='Time (ps)',
            yaxis_title='Count'
        )
        fig.show()

def get_data(folder, molecules_to_plot, plot_extra_variables, 
                num_molecules_to_plot, num_extra_variables_to_plot, 
                num_to_plot, MD_or_KMC, num):
    path = constants.result_path + folder
    
    if MD_or_KMC == 'MD':
        ending = 'MD'
        beginning = 'MD'
    else:
        ending = 'KMC_' + str(num)
        beginning = 'KMC'

    mol_per_frame = sparse.load_npz(path + 'Molecules_Per_Frame_' + ending \
                                    + '.npz').toarray()
    extra_variables = np.load(path + 'Extra_Variables_' + ending + '.npy')
    mol_list = np.load(path + 'Molecule_List_' + ending + '.npy')
    time_range = np.load(path + 'Time_Range_' + ending + '.npy')
    details = pkl.load(open(path + beginning + '_Details.pkl', 'rb'))
    extra_variables_tracked = details['Track_Extra_Variables']

    assert extra_variables_tracked.shape[0] == plot_extra_variables.shape[0], \
            "Not the same number of extra variables"

    mol_to_plot = np.zeros([time_range.shape[0], num_to_plot])

    for m in range(num_molecules_to_plot):
        mol = np.array(molecules_to_plot[m].split('-'), dtype=int)
        idx = np.where((mol_list == mol).all(-1))[0][0]
        mol_to_plot[:, m] = mol_per_frame[:, idx]

    for ev in range(num_extra_variables_to_plot):
        if extra_variables_tracked[ev] and plot_extra_variables[ev] == 1:
            mol_to_plot[:, ev + num_molecules_to_plot] = extra_variables[:, ev]

    if MD_or_KMC == 'MD':
        time_range = time_range*details['Timestep']
            
    return mol_to_plot, time_range

if __name__=='__main__':
    main()
