"""
Run subject-level models.

todo:
- smoothing: 5mm fwhm, determine brightness threshold from GM, WM, and
  outside brain signal. DONE.
- work around missing conditions (participant responses)
- detect outliers and add to design matrix. DONE.
"""
import os
import os.path as op
import sys
from glob import glob
from collections import Counter
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import nibabel as nib
from nipype.interfaces import fsl
from nilearn.masking import apply_mask
from nilearn.image import mean_img, concat_imgs, resample_to_img
from nistats.reporting import plot_design_matrix
from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel

# Local imports
import enhance_censoring as ec
from utils import get_run, get_subjects, calhoun_correction


def smooth_files(sub='sub-001'):
    """
    Use SUSAN to smooth functional data.
    """
    from nilearn.image import smooth_img
    fwhm = 5.0  # ~2x voxel size

    in_dir = '/scratch/kbott/narps/'
    fp_dir = op.join(in_dir, 'derivatives/fmriprep/')
    out_dir = op.join(in_dir, 'derivatives/fmriprep-smoothing/')
    sub_out_dir = op.join(out_dir, sub, 'func')
    os.makedirs(sub_out_dir, exist_ok=True)

    sub_func_dir = op.join(fp_dir, sub, 'func')
    func_files = sorted(glob(op.join(
        sub_func_dir, '*_task-MGT_*space-MNI152NLin2009cAsym_preproc.nii.gz')))

    print('Computing brightness threshold for smoothing')
    for func_file in func_files:
        run = get_run(func_file)

        # Run normal smoothing
        print('Smoothing data')
        smoothed_file = op.join(sub_out_dir,
                                '{0}_task-MGT_{1}_space-MNI152NLin2009cAsym_'
                                'desc-smoothed_bold.nii.gz'.format(sub, run))

        func_img = nib.load(func_file)
        smoothed_img = smooth_img(func_img, fwhm)
        smoothed_img.to_filename(smoothed_file)


def run_first_level(sub='sub-001'):
    """
    sub-003 is missing some conditions
    """
    # Settings for censoring
    fd_thresh = 0.9  # from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3895106/
    n_contig = 1
    n_before = 1
    n_after = 2

    # Settings for model
    hrf_model = 'spm + derivative + dispersion'
    period_cut = 128.
    drift_model = 'Cosine'
    slice_time_ref = 0.

    # Folders and files
    in_dir = '/scratch/kbott/narps/'
    nbc_dir = '/home/data/nbc/Laird_NARPS'
    fp_dir = op.join(in_dir, 'derivatives/fmriprep/')
    sm_dir = op.join(in_dir, 'derivatives/fmriprep-smoothing/')
    ev_dir = op.join(in_dir, 'event_tsvs')
    out_dir = op.join(nbc_dir, 'derivatives/first-levels')
    sub_out_dir = op.join(out_dir, sub)
    os.makedirs(sub_out_dir, exist_ok=True)

    out_gain_name = '{0}_task-MGT_space-MNI152NLin2009cAsym_desc-gain_betas.nii.gz'.format(sub)
    out_gain_file = op.join(sub_out_dir, out_gain_name)
    out_loss_name = '{0}_task-MGT_space-MNI152NLin2009cAsym_desc-loss_betas.nii.gz'.format(sub)
    out_loss_file = op.join(sub_out_dir, out_loss_name)

    # Find files
    sub_func_dir = op.join(fp_dir, sub, 'func')
    sub_sm_dir = op.join(sm_dir, sub, 'func')
    smoothed_files = sorted(glob(op.join(
        sub_sm_dir, '*_task-MGT_*space-MNI152NLin2009cAsym_desc-smoothed_bold.nii.gz')))
    brainmask = sorted(glob(op.join(
        sub_func_dir, '*_task-MGT_*space-MNI152NLin2009cAsym_brainmask.nii.gz')))[0]
    cf_files = sorted(glob(op.join(sub_func_dir, '*_task-MGT_*bold_confounds.tsv')))
    ev_files = sorted(glob(op.join(ev_dir, '{0}*.tsv'.format(sub))))

    # Check files to validate lazy searches
    assert len(smoothed_files) == len(cf_files) == len(ev_files)
    assert all([get_run(smoothed_files[i]) == get_run(cf_files[i]) == get_run(ev_files[i])
                for i in range(len(smoothed_files))])

    # Get parameters for modeling
    img = nib.load(smoothed_files[0])
    # Don't worry, this *is* 1 sec. Not a header problem.
    tr = img.header.get_zooms()[-1]

    fmri_imgs = [nib.load(func_file) for func_file in smoothed_files]
    affine, shape = fmri_imgs[0].affine, fmri_imgs[0].shape

    print('Generating design matrices')
    design_matrices = []
    for i_run, img in enumerate(fmri_imgs):
        n_vols = img.shape[-1]
        frame_times = np.arange(n_vols) * tr

        # Load in confounds
        cf_df = pd.read_csv(cf_files[i_run], sep='\t')
        mot_df = cf_df[['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']]
        mot_df = mot_df.add_prefix('motion_')
        fd = cf_df['FramewiseDisplacement'].values
        fd[0] = 0.  # replace nan
        cens_vec = ec.censor(fd, fd_thresh, n_contig=0,
                             n_before=n_before, n_after=n_after)

        # Check for non-steady state volumes
        nss_outlier_cols = [c for c in cf_df.columns if c.startswith('NonSteadyStateOutlier')]
        if nss_outlier_cols:
            nss_outliers_arr = cf_df[nss_outlier_cols].values
            nss_outliers_vec = np.sum(nss_outliers_arr, axis=1)
            cens_vec[nss_outliers_vec == 1] = 1

        cens_vec = ec.enhance(cens_vec, n_contig=n_contig,
                              n_before=0, n_after=0)
        cens_arr = ec._to_arr(cens_vec)

        # Build experimental paradigm
        ev_df = pd.read_csv(ev_files[i_run], sep='\t')
        ev_df['trial_type'] = ev_df['participant_response'].map({'strongly_reject': 'response',
                                                                 'weakly_reject': 'response',
                                                                 'weakly_accept': 'response',
                                                                 'strongly_accept': 'response',
                                                                 'NoResp': 'no_response'})
        ev_df.loc[ev_df['RT'] == 0, 'RT'] = ev_df['duration']
        ev_df['duration'] = ev_df['RT']
        ev_df['response'] = ev_df['participant_response'].map({'strongly_reject': 0,
                                                               'weakly_reject': 1,
                                                               'weakly_accept': 2,
                                                               'strongly_accept': 3,
                                                               'NoResp': 0})

        ev_df_gain = ev_df.copy()
        ev_df_loss = ev_df.copy()
        ev_df_resp = ev_df.copy()
        # Mean center (necessary) and variance normalize (maybe useful?) modulators
        ev_df_gain['modulation'] = (ev_df['gain'] - \
                                    ev_df.loc[ev_df['trial_type'] == 'response', 'gain'].mean()) / \
                                   ev_df.loc[ev_df['trial_type'] == 'response', 'gain'].std()
        ev_df_loss['modulation'] = (ev_df['loss'] - \
                                    ev_df.loc[ev_df['trial_type'] == 'response', 'loss'].mean()) / \
                                   ev_df.loc[ev_df['trial_type'] == 'response', 'loss'].std()
        ev_df_resp['modulation'] = (ev_df['response'] - \
                                    ev_df.loc[ev_df['trial_type'] == 'response', 'response'].mean()) / \
                                   ev_df.loc[ev_df['trial_type'] == 'response', 'response'].std()

        # Generate modulator design matrices to grab regressors from
        reg_cols = ['response', 'response_derivative', 'response_dispersion']
        gain_names = [col+'*gain' for col in reg_cols]
        loss_names = [col+'*loss' for col in reg_cols]
        resp_names = [col+'*response' for col in reg_cols]
        mot_names = mot_df.columns.tolist()
        dm_gain = make_first_level_design_matrix(
            frame_times,
            ev_df_gain,
            hrf_model=hrf_model,
            period_cut=period_cut,
            drift_model=drift_model
            )
        dm_gain = dm_gain.add_suffix('*gain')
        dm_loss = make_first_level_design_matrix(
            frame_times,
            ev_df_loss,
            hrf_model=hrf_model,
            period_cut=period_cut,
            drift_model=drift_model
            )
        dm_loss = dm_loss.add_suffix('*loss')
        dm_resp = make_first_level_design_matrix(
            frame_times,
            ev_df_resp,
            hrf_model=hrf_model,
            period_cut=period_cut,
            drift_model=drift_model
            )
        dm_resp = dm_resp.add_suffix('*response')
        
        dm_mod = pd.concat((dm_gain, dm_loss, dm_resp, mot_df), axis=1)

        # make main effect design matrix with modulators added
        dm = make_first_level_design_matrix(
            frame_times,
            ev_df,
            hrf_model=hrf_model,
            period_cut=period_cut,
            drift_model=drift_model,
            add_regs=dm_mod[gain_names+loss_names+resp_names+mot_names],
            add_reg_names=gain_names+loss_names+resp_names+mot_names
            )

        for i_vol in range(cens_arr.shape[1]):
            dm['censor_{0}'.format(i_vol)] = cens_arr[:, i_vol]

        # this list should be comprehensive
        sort_order = ['response', 'no_response', 'motion', 'censor', 'drift', 'constant']
        columns = [[c for c in dm.columns if c.startswith(so)] for so in sort_order]
        columns = [sorted(cols) for cols in columns]
        columns = [v for sl in columns for v in sl]

        # Check that new list has same elements as old list (in diff order)
        assert Counter(dm.columns.tolist()) == Counter(columns)
        dm = dm[columns]
        
        # Save image of design matrix
        run_name = get_run(smoothed_files[i_run])
        dm_name = '{0}_task-MGT_{1}_designMatrix.png'.format(sub, run_name)
        dm_file = op.join(sub_out_dir, dm_name)
        fig, ax = plt.subplots(figsize=(20, 16))
        plot_design_matrix(dm, ax=ax)
        fig.savefig(dm_file, dpi=400)
        dm_name = '{0}_task-MGT_{1}_designMatrix.tsv'.format(sub, run_name)
        dm_file = op.join(sub_out_dir, dm_name)
        dm.to_csv(dm_file, sep='\t', index=False)

        # put the design matrices in a list
        design_matrices.append(dm)

    contrasts = {
        'response*gain': [],
        'response_derivative*gain': [],
        'response_dispersion*gain': [],
        'response*loss': [],
        'response_derivative*loss': [],
        'response_dispersion*loss': [],
        'response*response': [],
        'response_derivative*response': [],
        'response_dispersion*response': [],
        }

    print('Designing contrasts')
    # Must loop through and define each contrast as multiple arrays because of
    # varying numbers of columns per run (based on outlier volumes and NoResps)
    for dm in design_matrices:
        contrast_matrix = np.eye(dm.shape[1])
        basic_contrasts = dict([(column, contrast_matrix[i])
                                for i, column in enumerate(dm.columns)])
        contrasts['response*gain'].append(basic_contrasts['response*gain'])
        contrasts['response_derivative*gain'].append(basic_contrasts['response_derivative*gain'])
        contrasts['response_dispersion*gain'].append(basic_contrasts['response_dispersion*gain'])
        contrasts['response*loss'].append(basic_contrasts['response*loss'])
        contrasts['response_derivative*loss'].append(basic_contrasts['response_derivative*loss'])
        contrasts['response_dispersion*loss'].append(basic_contrasts['response_dispersion*loss'])
        contrasts['response*response'].append(basic_contrasts['response*response'])
        contrasts['response_derivative*response'].append(basic_contrasts['response_derivative*response'])
        contrasts['response_dispersion*response'].append(basic_contrasts['response_dispersion*response'])

    print('Fitting a GLM')
    fmri_glm = FirstLevelModel(t_r=tr, slice_time_ref=slice_time_ref,
                               hrf_model=hrf_model, drift_model=drift_model,
                               period_cut=period_cut, smoothing_fwhm=None,
                               mask=brainmask)
    fitted_glm = fmri_glm.fit(fmri_imgs, design_matrices=design_matrices)

    print('Computing contrasts')
    beta_maps = {}

    # Iterate on contrasts
    for contrast_id, contrast_val in contrasts.items():
        print("\tcontrast id: %s" % contrast_id)
        # compute the contrasts
        beta_maps[contrast_id] = fmri_glm.compute_contrast(
            contrast_val, output_type='effect_size')

    print('Combining contrasts into magnitude images')
    gain_mag = calhoun_correction(
        beta_maps['response*gain'],
        beta_maps['response_derivative*gain'],
        beta_maps['response_dispersion*gain'])
    loss_mag = calhoun_correction(
        beta_maps['response*loss'],
        beta_maps['response_derivative*loss'],
        beta_maps['response_dispersion*loss'])

    gain_mag.to_filename(out_gain_file)
    loss_mag.to_filename(out_loss_file)


if __name__ == '__main__':
    """
    Run from command line so we can submit a separate job
    for each subject
    """
    sub = sys.argv[1]
    if sub in get_subjects():
        print('Running {0}'.format(sub))
        smooth_files(sub)
        run_first_level(sub)
    else:
        raise ValueError('{0} not found in get_subjects'.format(sub))
