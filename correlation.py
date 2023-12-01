import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def plot_correlation(array1, array2, x_label, y_label, title):
    # Extract necessary information from arrays
    x_values = array1  # Assuming the first entry in vt_values is the relevant one
    y_values = array2

    # Compute correlation
    correlation, _ = pearsonr(x_values, y_values)

    # Plotting with regression line
    plt.figure()
    plt.plot(x_values, y_values, 'o', label='Data Points')

    # Fit a linear regression model
    model = LinearRegression()
    x_values = x_values.reshape(-1, 1)
    model.fit(x_values, y_values)

    # Plot the regression line
    plt.plot(x_values, model.predict(x_values), color='red', linewidth=2,
             label='Regression Line')

    plt.title(f'{title}\nCorrelation: {correlation.item():.2f}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def compute_mean_vt_over_ROIs(trueplasma_directory, signaldata_directory):
    trueplasma_files = [file for file in os.listdir(trueplasma_directory) if file.endswith('_TRUEPLASMA.txt')]
    signaldata_files = [file for file in os.listdir(signaldata_directory) if file.endswith('_SIGNALDATA.txt')]

    correlations = []

    # Define a structured array for trueplasma data
    dt_trueplasma = np.dtype([('subject_id', int), ('roi', 'U10'), ('vt_values', object)])
    trueplasma_array = np.array([], dtype=dt_trueplasma)

    # Define a structured array for signaldata data
    dt_signaldata = np.dtype([('subject_id', int), ('roi', 'U10'), ('vt_values', object)])
    signaldata_array = np.array([], dtype=dt_signaldata)

    for trueplasma_file in trueplasma_files:
        subject_id = int(trueplasma_file.split('_')[0])
        roi = trueplasma_file.split('_')[1]

        # Find corresponding SIGNALDATA file
        pattern = f'{subject_id}_{roi}'
        matching_signaldata_file = [file for file in signaldata_files if pattern in file]
        matching_trueplasma_file = [file for file in trueplasma_files if pattern in file]

        if len(matching_signaldata_file) > 0:
            # Take the first matching SIGNALDATA file (you may need to adjust this logic)
            signaldata_file = matching_signaldata_file[0]
            trueplasma_file = matching_trueplasma_file[0]

            trueplasma_path = os.path.join(trueplasma_directory, trueplasma_file)
            signaldata_path = os.path.join(signaldata_directory, signaldata_file)

            trueplasma_vt_all = np.loadtxt(trueplasma_path)
            signaldata_vt_all = np.loadtxt(signaldata_path)

            # Append data to the trueplasma array
            entry_trueplasma = np.array([(subject_id, roi, trueplasma_vt_all)], dtype=dt_trueplasma)
            trueplasma_array = np.concatenate((trueplasma_array, entry_trueplasma))

            # Append data to the signaldata array
            entry_signaldata = np.array([(subject_id, roi, signaldata_vt_all)], dtype=dt_signaldata)
            signaldata_array = np.concatenate((signaldata_array, entry_signaldata))

    # Calculate average Vt values over ROIs for trueplasma
    unique_subjects = np.unique(trueplasma_array['subject_id'])

    trueplasma_avg_array = np.array([], dtype=dt_trueplasma)
    signaldata_avg_array = np.array([], dtype=dt_signaldata)

    for subject_id in unique_subjects:
        subject_entries = trueplasma_array[trueplasma_array['subject_id'] == subject_id]
        avg_vt_values = np.mean(np.vstack(subject_entries['vt_values']), axis=0)
        entry_avg_trueplasma = np.array([(subject_id, 'avg', avg_vt_values)], dtype=dt_trueplasma)
        trueplasma_avg_array = np.concatenate((trueplasma_avg_array, entry_avg_trueplasma))

        subject_entries = signaldata_array[signaldata_array['subject_id'] == subject_id]
        avg_vt_values = np.mean(np.vstack(subject_entries['vt_values']), axis=0)
        entry_avg_signaldata = np.array([(subject_id, 'avg', avg_vt_values)], dtype=dt_signaldata)
        signaldata_avg_array = np.concatenate((signaldata_avg_array, entry_avg_signaldata))

    # Access data in a structured way, e.g., trueplasma_avg_array['subject_id'], trueplasma_avg_array['vt_values'], etc.
    print("Trueplasma Averaged Data:")
    print(trueplasma_avg_array)

    print("\nSignaldata Averaged Data:")
    print(signaldata_avg_array)

    return trueplasma_avg_array['vt_values'], signaldata_avg_array['vt_values']


# Specify the directories for TRUEPLASMA and SIGNALDATA outputs
trueplasma_output_directory = '/Users/luto/Dropbox/AIProject/OUT/TRUEPLASMA/'
signaldata_output_directory = '/Users/luto/Dropbox/AIProject/OUT/SIGNALDATA/'

# Call the function to compute and print the correlation with plotting
trueplasma, signaldata =compute_mean_vt_over_ROIs(trueplasma_output_directory, signaldata_output_directory)

plot_correlation(trueplasma, signaldata, 'TRUEPLASMA Vt', 'SIGNALDATA Vt', 'Correlation between TRUEPLASMA and SIGNALDATA')
