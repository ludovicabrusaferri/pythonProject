import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz


def calculate_logan_vt(frame_time_filename, reference_filename, target_filename, tstar, output_filename, output_directory, reference_name):
    # Load data
    time_all = np.loadtxt(frame_time_filename, delimiter='\t')
    reference_tac = np.loadtxt(reference_filename)
    target_tac = np.loadtxt(target_filename)

    # Transpose if target is a row vector
    if target_tac.shape[0] == 1:
        target_tac = target_tac.T

    # Time calculations
    time = (time_all[:, 1] / 60 + time_all[:, 0] / 60) / 2
    dt = time_all[:, 1] / 60 - time_all[:, 0] / 60

    # Calculate integrated reference and target
    int_ref = cumtrapz(reference_tac, time, initial=0)
    int_target = cumtrapz(target_tac, time, initial=0)
    intercept = np.ones_like(reference_tac)

    # Construct matrices for linear regression
    X_matrix = np.vstack([(int_ref / target_tac).T, intercept]).T
    Y_vector = (int_target / target_tac).T

    # Select data after tstar
    tstar_index = np.min(np.where(time >= tstar))
    X_selected = X_matrix[tstar_index:, :]
    Y_selected = Y_vector[tstar_index:]

    # Weight matrix
    weight_matrix = np.diag(np.ones_like(dt))
    weight_matrix_selected = weight_matrix[tstar_index:, tstar_index:]

    # Perform linear regression
    regression_coefficients = np.linalg.lstsq(weight_matrix_selected @ X_selected, weight_matrix_selected @ Y_selected, rcond=None)[0]
    Vt = regression_coefficients[0]

    # Check for NaN in Vt
    if np.isnan(Vt):
        raise ValueError('Vt is NaN. Check input values!')

    # Plot and save figures
    print('Making the figures now..')
    plt.figure()
    plt.plot(X_matrix[:, 0], Y_vector, '*', X_selected[:, 0], X_selected @ regression_coefficients, 'k')
    plt.plot(X_matrix[tstar_index, 0], Y_vector[tstar_index], 'o', markersize=10)
    plt.title(f'Logan Vt: target-{reference_name}, reference-plasma, tstar: {tstar} minutes')
    plt.ylabel('Y')
    plt.xlabel('X')

    # Set equal scaling for both axes
    plt.axis('equal')

    plt.grid(True)
    plt.savefig(output_filename, format='png')
    plt.close()
    print('Figures DONE..')

    # Save Vt to a text file
    with open(f'{output_filename}.txt', 'w') as fid:
        fid.write(f'{Vt}\n')

    # Move file to the specified output directory
    if output_directory:
        os.rename(f'{output_filename}.txt', os.path.join(output_directory, f'{output_filename}.txt'))


# Main Script
script_directory = '/Users/luto/Dropbox/AIProject/ScriptsAI/PETkinetic'
os.chdir(script_directory)

start_value = 0
N = 2
subject_ids = list(range(start_value, N))
print(subject_ids)
subjects = [str(i) for i in subject_ids]
tstar = 15
filename_option = 'TRUEPLASMA'

home_directory = os.path.abspath(script_directory)
home_directory = home_directory.replace('/ScriptsAI/PETkinetic', '')

tac_directory = '/Users/luto/Dropbox/AIProject/DATA/TAC'
output_directory = '/Users/luto/Dropbox/AIProject/OUT/'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

os.chdir(home_directory)


if filename_option == 'TRUEPLASMA':
    path_to_use = os.path.join(home_directory, 'DATA/metabolite_corrected_signal_data')
else:
    print("Invalid filename_option. Exiting.")
    sys.exit()

for i in range(len(subject_ids)):
    print(f'============ SUBJ {subjects[i]} and i: {i} \n\n =======\n')
    if os.path.exists(os.path.join(path_to_use, f'{subjects[i]}.txt')):
        frame_time_filename = os.path.join(home_directory, 'DATA/Analyses/', 'timeframes_start_stop.txt')
        plasma_Lin_filename = os.path.join(path_to_use, f'{subjects[i]}.txt')

        os.chdir(tac_directory)
        tacs = [tac for tac in os.listdir('.') if tac.startswith(f'{subjects[i]}') and tac.endswith('_SUV-TAC.txt')]

        for tac in tacs:
            tac_name = tac.split('_SUV')[0]
            reference_name = tac_name.replace(f'{subjects[i]}_', '')
            ref_filename = plasma_Lin_filename

            if not os.path.exists(os.path.join(output_directory, f'{tac_name}_Logan_Vt_{tstar}min')):
                calculate_logan_vt(frame_time_filename, ref_filename, tac, tstar,
                                   os.path.join(output_directory, f'{tac_name}_Logan_Vt_{tstar}min_{filename_option}'),
                                   output_directory, reference_name)
