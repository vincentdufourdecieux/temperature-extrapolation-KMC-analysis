Code associated with the paper: Dufour-Décieux, V., Ransom, B., Sendek, A. D., Freitas, R., Blanchet, J., & Reed, E. J. (2022). Temperature Extrapolation of Molecular Dynamics Simulations of Complex Chemistry to Microsecond Timescales Using Kinetic Models: Applications to Hydrocarbon Pyrolysis. Journal of Chemical Theory and Computation (https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00623).

**Code organization**

The code is organized as follows:
- folder 'data': 
    - 'MD_data' folder: the '.atom' files of the MD simulation should be added in this folder. There are already 4 files that will be used in the 'First test analyses' section (see below). A '.atom' file is the output type of the file used by LAMMPS (https://docs.lammps.org/dump.html). Examples of these files are provided in data/MD_data. 
    - 'Results' folder: that is where the results of the mechanism extraction or the KMC run will be saved. 
    - 'Results_ref_comparison' folder: this contains the reference results for comparison with your results after 'First test analyses' (see below)
- 'features' folder: featurization of atoms, of molecules, and of reactions.
- 'MD_analysis' folder: extraction of mechanism of a MD file.
- 'KMC_simulation' folder: run the KMC simulations
- 'plot' folder: plot the results
- 'util' folder: save the data or constant variables definitions
- main.py: all the information for the analyses will be here. Many variables can be changed here.
- 'requirements.txt': packages required to run this code.


**First test analyses**
- Run main.py as it is in this Github repository. This will run a mechanism extraction and one KMC from this mechanism for the 4 '.atom' files that are in data/MD_data. This should take several minutes. 
- Compare the results files with the files in data/Resuts_ref_comparison/. For example, check that data/Results/3100K_1_120ps/Analysis/Reactions.csv gives the same results as data/Results_ref_comparison/3100K_1_120ps/Analysis/Reactions.csv.
- To run a temperature extrapolation, in main.py, perform the following changes:
    - change temperatures (line 8) from [3100, 3200, 3300, 3400] to [3400]
    - change MD_analysis_bool (line 12) from 1 to 0 
    - change foldername_reactions (line 35) from [str(temperatures[j]) + 'K_1_120ps/'] to ['3100K_1_120ps/', '3200K_1_120ps/', '3300K_1_120ps/']
    - change foldername_save_KMC (line 37) from str(temperatures[j]) + 'K_1_120ps/KMC/' to 'temp_extrap_to_3400K/KMC/'
    - Run main.py. This should take less than one minute.
- This is running a temperature extrapolation from the mechanism extracted at 3100, 3200, and 3300K to 3400K. You can compare the files in data/Results/temp_extrap_to_3400K/, they should be the same as in data/Results_ref_comparison/temp_extrap_to_3400K if you kept the seed 'a = 1' (line 67).  
