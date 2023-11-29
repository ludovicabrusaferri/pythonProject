import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

def PBR_Logan_ROI(frame_time_fn, reference_fn, target_fn, tstar, output_fn, outdir, reference_name):
    time_all = np.loadtxt(frame_time_fn, delimiter='\t')
    ref = np.loadtxt(reference_fn)
    target = np.loadtxt(target_fn)

    if target.shape[0] == 1:
        target = target.T

    time = (time_all[:, 1]/60 + time_all[:, 0]/60) / 2
    dt = time_all[:, 1]/60 - time_all[:, 0]/60

    int_ref = cumtrapz(ref, time, initial=0)
    int_target = cumtrapz(target, time, initial=0)
    intercept = np.ones_like(ref)

    X = np.vstack([(int_ref / target).T, intercept]).T
    Y = (int_target / target).T

    tstar_index = np.min(np.where(time >= tstar))
    XX = X[tstar_index:, :]
    YY = Y[tstar_index:]

    wt = np.diag(np.ones_like(dt))
    wtwt = wt[tstar_index:, tstar_index:]

    p = np.linalg.lstsq(wtwt @ XX, wtwt @ YY, rcond=None)[0]
    Vt = p[0]

    if np.isnan(Vt):
        raise ValueError('Vt is NaN. Check input values!')

    print('Making the figures now..')
    plt.figure()
    plt.plot(X[:, 0], Y, '*', XX[:, 0], XX @ p, 'k')
    plt.plot(X[tstar_index, 0], Y[tstar_index], 'o', markersize=10)
    plt.title(f'Logan Vt: target-{reference_name}, reference-plasma, tstar: {tstar} minutes')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.grid(True)
    plt.savefig(output_fn, format='png')
    plt.close()
    print('Figures DONE..')

    with open(f'{output_fn}.txt', 'w') as fid:
        fid.write(f'{Vt}\n')

    if outdir:
        os.rename(f'{output_fn}.txt', os.path.join(outdir, f'{output_fn}.txt'))

def shorten_plasma(frame_time_fn, plasma_full, subj, homedir):
    filename = os.path.join(homedir, 'SUV-info_for_plasma.xls')
    SUV = np.genfromtxt(filename, delimiter='\t', skip_header=1)

    subj_list = SUV[:, 0].astype(str)
    subjs = subj_list[~np.isnan(subj_list)]
    index = np.where(subjs == subj)[0][0]

    subj_SUV = SUV[index - 1, 3]  # Subtract one because column headers are included

    time = np.loadtxt(frame_time_fn)
    plasma_met = np.loadtxt(plasma_full)
    plas_short = np.zeros_like(time[:, 0])

    plasmet = plasma_met
    plastime = plasmet[:, 0]
    tstart = time[0, 0] + 1
    pstart = tstart + 1

    for m in range(len(time[:, 0])):
        tend = time[m, 1]

        if tend >= plastime[-1]:
            tend = plastime[-1] - 1

        for x in range(len(plasmet[:, 0])):
            if plasmet[x, 0] == tend:
                pend = plasmet[x, 0] + 1

        plas_short[m, 0] = np.mean(plasmet[pstart:pend, 1])

        pstart = pend + 1

    suv_plasma = plas_short / subj_SUV
    short_plasma = suv_plasma

    if np.isnan(short_plasma):
        raise ValueError('There are not enough blood samples - it is likely that they do not cover the whole scan time!')

# Main Script
path = '/Users/luto/Dropbox/AIProject/ScriptsAI/PETkinetic'
os.chdir(path)

start = 0
N = 51
vec = [0]
subj = [str(i) for i in vec]

usethisfilename = 'TRUEPLASMA'

homedir = os.path.abspath(path)
homedir = homedir.replace('/ScriptsAI/PETkinetic', '')

TACdir = '/Users/luto/Dropbox/AIProject/DATA/TAC'
outdir = '/Users/luto/Dropbox/AIProject/OUT/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

os.chdir(homedir)

tstar = 15

TRUEPLASMA = os.path.join(homedir, 'DATA/metabolite_corrected_signal_data')

usethis = TRUEPLASMA

for i in range(len(vec)):
    print(f'============ SUBJ {subj[i]} and i: {i} \n\n =======\n')
    if os.path.exists(os.path.join(usethis, f'{subj[i]}.txt')):
        frame_time_fn = os.path.join(homedir, 'DATA/Analyses/', 'timeframes_start_stop.txt')
        plasma_Lin = os.path.join(usethis, f'{subj[i]}.txt')

        os.chdir(TACdir)
        tacs = [tac for tac in os.listdir('.') if tac.startswith(f'{subj[i]}') and tac.endswith('_SUV-TAC.txt')]

        for tac in tacs:
            tacnm = tac.split('_SUV')[0]
            reference_name = tacnm.replace(f'{subj[i]}_', '')
            ref_fn = plasma_Lin

            if not os.path.exists(os.path.join(outdir, f'{tacnm}_Logan_Vt_{tstar}min')):
                PBR_Logan_ROI(frame_time_fn, ref_fn, tac, tstar,
                              os.path.join(outdir, f'{tacnm}_Logan_Vt_{tstar}min_{usethisfilename}'), outdir, reference_name)
