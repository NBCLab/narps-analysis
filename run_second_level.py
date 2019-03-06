"""
Run group-level models.

todo:
- parametric effect of gain per group
- parametric effect of loss per group
- two groups: equalRange and equalIndifference
- ROIs for vmPFC, ventral striatum (NAcc-harvox), & amygdala (harvox)
- group comparison for loss
"""
import os
import os.path as op
import sys
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from datetime import datetime
from nipype.interfaces.fsl import Merge, Randomise
from nistats.reporting import plot_design_matrix
from nistats.design_matrix import make_second_level_design_matrix

# Folders and files
dat_dir = '/home/data/nbc/Laird_NARPS/derivatives'
tf_dir = '/scratch/kbott/narps/'
out_gain_dir = op.join(dat_dir, 'group-level-gain')
out_loss_dir = op.join(dat_dir, 'group-level-loss')
out_diff_dir = op.join(dat_dir, 'group-level-diff')
gain_4d_betas = op.join(out_gain_dir, 'task-MGT_space-MNI152NLin2009cAsym_desc-gain_4d_betas.nii.gz')
loss_4d_betas = op.join(out_loss_dir, 'task-MGT_space-MNI152NLin2009cAsym_desc-loss_4d_betas.nii.gz')
dm_file = op.join(tf_dir, 'event_tsvs', 'participants.tsv')

loss_mat = op.join(out_loss_dir, 'design.mat')
loss_con = op.join(out_loss_dir, 'design.con')

gain_mat = op.join(out_gain_dir, 'design.mat')
gain_con = op.join(out_gain_dir, 'design.con')

# Read in the participants df
ppts = pd.read_csv(dm_file, sep='\t', index_col=0, header=0)
ppt_dummy = ppts.replace({'gender':{'M':0, 'F':1}})
ppt_dummy['equalRange'] = ppt_dummy['group']
ppt_dummy['equalIndifference'] = ppt_dummy['group']
ppt_dummy.replace({'equalRange': {'equalIndifference':0, 'equalRange':1},
                   'equalIndifference': {'equalIndifference':1, 'equalRange':0}},
                   inplace=True)
ppt_dummy.drop('group', axis=1, inplace=True)

# Make a list of beta map files for the gain group level analysis
gain_betas = []
loss_betas = []
for sub in ppts.index.values:
    try:
        sub_fl_dir = op.join(dat_dir, 'first-levels', sub)
        gain_name = '{0}_task-MGT_space-MNI152NLin2009cAsym_desc-gain_betas.nii.gz'.format(sub)
        gain_file = op.join(sub_fl_dir, gain_name)
        gain_betas.append(gain_file)

        loss_name = '{0}_task-MGT_space-MNI152NLin2009cAsym_desc-loss_betas.nii.gz'.format(sub)
        loss_file = op.join(sub_fl_dir, loss_name)
        loss_betas.append(loss_file)

        #And while we're iterating over subjects, calculate avg FramewiseDisplacement
        sub_fp_dir = op.join(tf_dir, 'derivatives/fmriprep', sub, 'func')
        sub_fd = 0
        for run in ['01', '02', '03', '04']:
            cf_file = op.join(sub_fp_dir, '{0}_task-MGT_run-{1}_bold_confounds.tsv'.format(sub, run))
            cf = pd.read_csv(cf_file, sep='\t', index_col=0, header=0)
            sub_fd += np.mean(cf['FramewiseDisplacement'])
        ppt_dummy.at[sub, 'fd'] = sub_fd
    except Exception as e:
        print(e)
        print('Could not find files for {0}, droppped {0} from analysis'.format(sub))
        ppt_dummy.drop(sub, axis=0, inplace=True)

# Confounds for group-level analyses
confounds = ['fd', 'gender', 'age']
groups = ['equalRange', 'equalIndifference']

ppt_dummy = ppt_dummy[['equalRange', 'equalIndifference', 'fd', 'gender', 'age']]

# Mean center the confounding variables...
#for confound in confounds:
#    mean = np.mean(ppt_dummy[confound])
#    ppt_dummy['{0}_mc'.format(confound)] = ppt_dummy[confound] - mean
#    ppt_dummy.drop(confound, axis=1, inplace=True)

ppt_dummy['subject_label'] = ppt_dummy.index

# Make an fsl-compatible design matrix!
dmat = make_second_level_design_matrix(subjects_label=ppt_dummy.index.values, confounds=ppt_dummy)
dmat.to_csv(op.join(out_loss_dir, 'design_matrix.txt'), sep='\t')
dmat.to_csv(op.join(out_gain_dir, 'design_matrix.txt'), sep='\t')

fig,ax = plt.subplots(figsize=(10, 7))
g = plot_design_matrix(dmat, ax=ax)
ax.set_title('Second level design matrix', fontsize=12)
ax.set_ylabel('maps')
plt.tight_layout()
fig.savefig(op.join(out_gain_dir, 'design_matrix.png'), dpi=300)
fig.savefig(op.join(out_loss_dir, 'design_matrix.png'), dpi=300)


for mat in [gain_mat, loss_mat]:
    covariates = []
    dmat_file = open(mat, "w+")
    dmat_file.write('/NumWaves\t{0}\n/NumPoints\t{1}\n'.format(len(dmat.keys())-1, len(dmat.index)))
    dmat_file.write('/PPheights\t\t1.000000e+00\t1.000000e+00\t0.000000e+00\t0.000000e+00\t0.000000e+00\n')
    dmat_file.write('\n/Matrix\n')
    for i in dmat.index:
        dmat_file.writelines('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(dmat.iloc[i, 0],
                                                              dmat.iloc[i, 1],
                                                              dmat.iloc[i, 2],
                                                              dmat.iloc[i, 3],
                                                              dmat.iloc[i, 4]))
        covariates.append(dmat.iloc[i] for i in dmat.index)
    dmat_file.close()

# And an fsl-compatible contrast file!
gain_cons = {'equalRange mean': '1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00',
             'equalIndifference mean': '0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00'}
loss_cons = {'equalRange > equalIndifference': '1.000000e+00 -1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00',
             'equalIndifference > equalRange': '-1.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00',
             'equalRange mean': '1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 ',
             'equalIndifference mean': '0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 '}
print('making contrasts!')
con_file = open(gain_con, "w+")
for i in np.arange(0, len(gain_cons.keys())):
    con_file.write('/ContrastName{0}\t{1}\n'.format(i, list(gain_cons.keys())[i]))
con_file.write('/NumWaves\t{0}\n'.format(len(dmat.keys())-1))
con_file.write('/NumContrasts\t{0}\n'.format(len(gain_cons.keys())))
con_file.write('/PPheights\t')
for i in np.arange(0, len(gain_cons.keys())):
    con_file.write('\t1.000000e+00')
con_file.write('\n/RequriedEffect\t\t0.515\t0.558')
con_file.write('\n\n/Matrix')
for i in np.arange(0, len(gain_cons.keys())):
    con_file.write('\n{0}'.format(list(gain_cons.values())[i]))
con_file.close()

con_file = open(loss_con, "w+")
for i in np.arange(0, len(loss_cons.keys())):
    con_file.write('/ContrastName{0}\t{1}\n'.format(i, list(loss_cons.keys())[i]))
con_file.write('/NumWaves\t{0}\n'.format(len(dmat.keys())-1))
con_file.write('/NumContrasts\t{0}\n'.format(len(loss_cons.keys())))
con_file.write('/PPheights\t')
for i in np.arange(0, len(loss_cons.keys())):
    con_file.write('\t1.000000e+00')
con_file.write('\n/RequiredEffect\t\t0.724\t0.724\t0.512\t0.512')
con_file.write('\n\n/Matrix')
for i in np.arange(0, len(loss_cons.keys())):
    con_file.write('\n{0}'.format(list(loss_cons.values())[i]))
con_file.close()

print('merging betas!')
# Convert a list of filenames into a 4D betas Nifti
merger = Merge(dimension='t', output_type='NIFTI_GZ')
gains = merger.run(in_files=gain_betas, merged_file=gain_4d_betas)
losses = merger.run(in_files=loss_betas, merged_file=loss_4d_betas)

# Perform sanity check: do design matrices match length of 4d betas?
print('loss betas: {0}\ngain betas: {1}\ndmat subjects: {2}\n.mat length: {3}'.format(len(loss_betas), len(gain_betas), len(dmat.index), len(covariates)))
assert len(gain_betas) == len(dmat.index), 'number of gain beta maps does not equal number of subjects in nistats design matrix'
assert len(loss_betas) == len(dmat.index), 'number of loss beta maps does not equal number of subjects in nistats design matrix'
assert len(gain_betas) == len(covariates), 'number of gain beta maps does not equal number of subjects in design.mat'
assert len(loss_betas) == len(covariates), 'number of loss beta maps does not equal number of subjects in design.mat'

# Run the models, nonparametrically, using Randomise
random = Randomise(num_perm=10000, cm_thresh=2.3, demean=True)
pre_rand = datetime.now()
print('{0}: about to randomise!'.format(pre_rand))
random.run(in_file=loss_4d_betas, tcon=loss_con, design_mat=loss_mat, base_name=op.join(out_loss_dir, 'task-MGT_space-MNI152NLin2009cAsym_desc-loss_randomise'))
loss_time = datetime.now() - pre_rand
pre_rand = datetime.now()
print('{0}: losses done! took {1} s.'.format(pre_rand, loss_time))
random.run(in_file=gain_4d_betas, tcon=gain_con, design_mat=gain_mat, base_name=op.join(out_gain_dir, 'task-MGT_space-MNI152NLin2009cAsym_desc-gain_randomise'))
gain_time = datetime.now() - pre_rand
print('{0}: losses done! took {1} s.'.format(datetime.now(), gain_time))
