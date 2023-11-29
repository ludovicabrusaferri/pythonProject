import os
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


def shorten_plasma(frame_time_filename, plasma_full, subject_id, home_directory):
    # Load SUV data
    suv_filename = os.path.join(home_directory, 'SUV-info_for_plasma.xls')
    suv_data = np.genfromtxt(suv_filename, delimiter='\t', skip_header=1)

    # Extract subject list and find index for the given subject_id
    subject_list = suv_data[:, 0].astype(str)
    subjects = subject_list[~np.isnan(subject_list)]
    subject_index = np.where(subjects == subject_id)[0][0]

    # Extract SUV value for the subject
    subject_suv = suv_data[subject_index - 1, 3]  # Subtract one because column headers are included

    # Load time and plasma data
    time_data = np.loadtxt(frame_time_filename)
    plasma_metabolite_data = np.loadtxt(plasma_full)
    plasma_shortened = np.zeros_like(time_data[:, 0])

    plasma_time = plasma_metabolite_data[:, 0]
    tstart = time_data[0, 0] + 1
    pstart = tstart + 1

    for m in range(len(time_data[:, 0])):
        tend = time_data[m, 1]

        if tend >= plasma_time[-1]:
            tend = plasma_time[-1] - 1

        for x in range(len(plasma_metabolite_data[:, 0])):
            if plasma_metabolite_data[x, 0] == tend:
                pend = plasma_metabolite_data[x, 0] + 1

        plasma_shortened[m, 0] = np.mean(plasma_metabolite_data[pstart:pend, 1])

        pstart = pend + 1

    suv_plasma = plasma_shortened / subject_suv
    shortened_plasma = suv_plasma

    # Check for NaN in shortened_plasma
    if np.isnan(shortened_plasma):
        raise ValueError('There are not enough blood samples - it is likely that they do not cover the whole scan time!')


# Main Script
script_directory = '/Users/luto/Dropbox/AIProject/ScriptsAI/PETkinetic'
os.chdir(script_directory)

start = 0
N = 51
subject_ids = [0]
subjects = [str(i) for i in subject_ids]

filename_option = 'TRUEPLASMA'

home_directory = os.path.abspath(script_directory)
home_directory = home_directory.replace('/ScriptsAI/PETkinetic', '')

tac_directory = '/Users/luto/Dropbox/AIProject/DATA/TAC'
output_directory = '/Users/luto/Dropbox/AIProject/OUT/'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

os.chdir(home_directory)

tstar = 15

TRUEPLASMA_path = os.path.join(home_directory, 'DATA/metabolite_corrected_signal_data')

filename_to_use = TRUEPLASMA_path

for i in range(len(subject_ids)):
    print(f'============ SUBJ {subjects[i]} and i: {i} \n\n =======\n')
    if os.path.exists(os.path.join(filename_to_use, f'{subjects[i]}.txt')):
        frame_time_filename = os.path.join(home_directory, 'DATA/Analyses/', 'timeframes_start_stop.txt')
        plasma_Lin_filename = os.path.join(filename_to_use, f'{subjects[i]}.txt')

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
