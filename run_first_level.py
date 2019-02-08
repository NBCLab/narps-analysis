"""
Run subject-level models.

todo:
- smoothing: 5mm fwhm, determine brightness threshold from GM, WM, and
  outside brain signal. DONE.
- work around missing conditions (participant responses)
- detect outliers and add to design matrix. DONE.
"""
import re
import os
import os.path as op
from glob import glob
from collections import Counter

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
import utils


def get_subjects():
    in_dir = '/scratch/kbott/narps/'
    fp_dir = op.join(in_dir, 'derivatives/fmriprep/')
    subjects = sorted([op.basename(op.splitext(f)[0]) for f in
                       glob(op.join(in_dir, 'derivatives/fmriprep/*.html'))])
    return subjects


def get_run(f):
    return re.findall('(run-[0-9]+)_', f)[0]


def smooth_files(sub='sub-001'):
    """
    Use SUSAN to smooth functional data.
    """
    fwhm = 5.0  # ~2x voxel size

    in_dir = '/scratch/kbott/narps/'
    fp_dir = op.join(in_dir, 'derivatives/fmriprep/')
    ev_dir = op.join(in_dir, 'event_tsvs')
    out_dir = op.join(in_dir, 'derivatives/fmriprep-smoothing/')
    sub_out_dir = op.join(out_dir, sub, 'func')
    os.makedirs(sub_out_dir, exist_ok=True)

    sub_anat_dir = op.join(fp_dir, sub, 'anat')
    sub_func_dir = op.join(fp_dir, sub, 'func')
    func_files = sorted(glob(op.join(
        sub_func_dir, '*_task-MGT_*space-MNI152NLin2009cAsym_preproc.nii.gz')))
    brainmask = sorted(glob(op.join(
        sub_func_dir, '*_task-MGT_*space-MNI152NLin2009cAsym_brainmask.nii.gz')))[0]

    print('Resampling tissue probability maps to functional resolution')
    gm_file = sorted(glob(op.join(
        sub_anat_dir, '*_space-MNI152NLin2009cAsym_class-GM_probtissue.nii.gz')))[0]
    gm_img = resample_to_img(gm_file, func_files[0], 'linear')  # same res for all funcs
    gm_prob = gm_img.get_data()
    gm_mask = (gm_prob > 0.5).astype(int)
    gm_img = nib.Nifti1Image(gm_mask, gm_img.affine)

    wm_file = sorted(glob(op.join(
        sub_anat_dir, '*_space-MNI152NLin2009cAsym_class-WM_probtissue.nii.gz')))[0]
    wm_img = resample_to_img(wm_file, func_files[0], 'linear')  # same res for all funcs
    wm_prob = wm_img.get_data()
    wm_mask = (wm_prob > 0.5).astype(int)
    wm_img = nib.Nifti1Image(wm_mask, wm_img.affine)

    bm_img = nib.load(brainmask)
    bm_mask = bm_img.get_data()
    ob_mask = 1 - bm_mask  # reverse (1 = outside brain, 0 = inside brain)
    ob_img = nib.Nifti1Image(ob_mask, bm_img.affine)

    print('Computing brightness threshold for smoothing')
    for func_file in func_files:
        run = get_run(func_file)
        gm_data = apply_mask(func_file, gm_img)
        mean_gm = np.mean(gm_data)

        wm_data = apply_mask(func_file, wm_img)
        mean_wm = np.mean(wm_data)

        mean_contrast = mean_gm - mean_wm

        ob_data = apply_mask(func_file, ob_img)
        mean_noise = np.mean(ob_data)

        # Brightness threshold halfway between contrast and noise
        brightness_threshold = np.mean((mean_contrast, mean_noise))

        print('Smoothing data')
        smoothed_file = op.join(sub_out_dir,
                                '{0}_task-MGT_{1}_space-MNI152NLin2009cAsym_'
                                'desc-smoothed_bold.nii.gz'.format(sub, run))

        sus = fsl.SUSAN()
        sus.inputs.in_file = func_files[run]
        sus.inputs.brightness_threshold = brightness_threshold
        sus.inputs.fwhm = fwhm
        sus.inputs.out_file = smoothed_file
        sus.run()


def run_first_level(sub='sub-001', mod='gain'):
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
    fp_dir = op.join(in_dir, 'derivatives/fmriprep/')
    sm_dir = op.join(in_dir, 'derivatives/fmriprep-smoothed/')
    ev_dir = op.join(in_dir, 'event_tsvs')
    out_dir = op.join(in_dir, 'derivatives/first-levels-{0}'.format(mod))
    sub_out_dir = op.join(out_dir, sub)
    os.makedirs(sub_out_dir, exist_ok=True)

    out_name = '{0}_task-MGT_space-MNI152NLin2009cAsym_desc-{1}_betas.nii.gz'.format(sub, mod)
    out_file = op.join(sub_out_dir, out_name)

    # Find files
    sub_func_dir = op.join(fp_dir, sub, 'func')
    sub_sm_dir = op.join(sm_dir, sub, 'func')
    smoothed_files = sorted(glob(op.join(
        sub_sm_dir, '*_task-MGT_*space-MNI152NLin2009cAsym_desc-smoothed_bold.nii.gz.nii.gz')))
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
        fd = cf_df['FramewiseDisplacement'].values
        fd[0] = 0  # replace nan
        cens_vec = ec.censor(fd, fd_thresh, n_contig=n_contig,
                             n_before=n_before, n_after=n_after)

        # Check for non-steady state volumes
        nss_outlier_cols = [c for c in cf_df.columns if c.startswith('NonSteadyStateOutlier')]
        if nss_outlier_cols:
            nss_outliers_arr = cf_df[nss_outlier_cols].values
            nss_outliers_vec = np.sum(nss_outliers_arr, axis=1)
            cens_vec[nss_outliers_vec == 1] = 1

        cens_arr = ec._to_arr(cens_vec)

        # Build experimental paradigm
        ev_df = pd.read_csv(ev_files[i_run], sep='\t')
        ev_df.loc[ev_df['RT'] == 0, 'RT'] = ev_df['duration']
        ev_df['duration'] = ev_df['RT']
        ev_df['trial_type'] = ev_df['participant_response']
        if i_run == 0:
            count_df = ev_df.groupby('trial_type').count()
        else:
            count_df += ev_df.groupby('trial_type').count()

        ev_df_mod = ev_df.copy()
        # Mean center (necessary) and variance normalize (maybe useful?) modulator
        ev_df_mod['modulation'] = (ev_df[mod] - ev_df[mod].mean()) / ev_df[mod].std()

        # Generate modulator design matrix to grab regressors from
        dm_pm = make_first_level_design_matrix(
            frame_times,
            ev_df_mod,
            hrf_model=hrf_model,
            period_cut=period_cut,
            drift_model=drift_model
            )
        cols = ev_df['participant_response'].unique()
        reg_cols = []
        for col in cols:
            reg_cols.append(col)
            reg_cols.append(col+'_derivative')
            reg_cols.append(col+'_dispersion')

        new_names = [col+'*'+mod for col in reg_cols]

        # make main effect design matrix with modulators added
        dm = make_first_level_design_matrix(
            frame_times,
            ev_df,
            hrf_model=hrf_model,
            period_cut=period_cut,
            drift_model=drift_model,
            add_regs=dm_pm[reg_cols],
            add_reg_names=new_names
            )

        for i_vol in range(cens_arr.shape[1]):
            dm['censor_{0}'.format(i_vol)] = cens_arr[:, i_vol]

        # this list should be comprehensive
        sort_order = ['weakly', 'strongly', 'NoResp', 'drift', 'censor', 'constant']
        columns = [[c for c in dm.columns if c.startswith(so)] for so in sort_order]
        columns = [sorted(cols) for cols in columns]
        columns = [v for sl in columns for v in sl]
        # Check that new list has same elements as old list (in diff order)
        assert Counter(dm.columns.tolist()) == Counter(columns)
        dm = dm[columns]

        # put the design matrices in a list
        design_matrices.append(dm)

    contrasts = {
        'strongly_reject*'+mod: [],
        'strongly_reject_derivative*'+mod: [],
        'strongly_reject_dispersion*'+mod: [],
        'weakly_reject*'+mod: [],
        'weakly_reject_derivative*'+mod: [],
        'weakly_reject_dispersion*'+mod: [],
        'strongly_accept*'+mod: [],
        'strongly_accept_derivative*'+mod: [],
        'strongly_accept_dispersion*'+mod: [],
        'weakly_accept*'+mod: [],
        'weakly_accept_derivative*'+mod: [],
        'weakly_accept_dispersion*'+mod: [],
        }

    print('Designing contrasts')
    # Must loop through and define each contrast as multiple arrays because of
    # varying numbers of columns per run (based on outlier volumes and NoResps)
    for dm in design_matrices:
        contrast_matrix = np.eye(dm.shape[1])
        basic_contrasts = dict([(column, contrast_matrix[i])
                                for i, column in enumerate(dm.columns)])
        contrasts['strongly_reject*'+mod].append(basic_contrasts['strongly_reject*'+mod])
        contrasts['strongly_reject_derivative*'+mod].append(basic_contrasts['strongly_reject_derivative*'+mod])
        contrasts['strongly_reject_dispersion*'+mod].append(basic_contrasts['strongly_reject_dispersion*'+mod])
        contrasts['weakly_reject*'+mod].append(basic_contrasts['weakly_reject*'+mod])
        contrasts['weakly_reject_derivative*'+mod].append(basic_contrasts['weakly_reject_derivative*'+mod])
        contrasts['weakly_reject_dispersion*'+mod].append(basic_contrasts['weakly_reject_dispersion*'+mod])
        contrasts['strongly_accept*'+mod].append(basic_contrasts['strongly_accept*'+mod])
        contrasts['strongly_accept_derivative*'+mod].append(basic_contrasts['strongly_accept_derivative*'+mod])
        contrasts['strongly_accept_dispersion*'+mod].append(basic_contrasts['strongly_accept_dispersion*'+mod])
        contrasts['weakly_accept*'+mod].append(basic_contrasts['weakly_accept*'+mod])
        contrasts['weakly_accept_derivative*'+mod].append(basic_contrasts['weakly_accept_derivative*'+mod])
        contrasts['weakly_accept_dispersion*'+mod].append(basic_contrasts['weakly_accept_dispersion*'+mod])

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
    strongly_accept_mag = utils.calhoun_correction(
        beta_maps['strongly_accept*'+mod],
        beta_maps['strongly_accept_derivative*'+mod],
        beta_maps['strongly_accept_dispersion*'+mod])
    weakly_accept_mag = utils.calhoun_correction(
        beta_maps['weakly_accept*'+mod],
        beta_maps['weakly_accept_derivative*'+mod],
        beta_maps['weakly_accept_dispersion*'+mod])
    weakly_reject_mag = utils.calhoun_correction(
        beta_maps['weakly_reject*'+mod],
        beta_maps['weakly_reject_derivative*'+mod],
        beta_maps['weakly_reject_dispersion*'+mod])
    strongly_reject_mag = utils.calhoun_correction(
        beta_maps['strongly_reject*'+mod],
        beta_maps['strongly_reject_derivative*'+mod],
        beta_maps['strongly_reject_dispersion*'+mod])

    print('Averaging across participant responses')
    prop_df = count_df['participant_response'][
        ['strongly_accept', 'weakly_accept', 'strongly_reject', 'weakly_reject']
    ]
    prop_df /= prop_df.sum()
    props = prop_df[['strongly_accept', 'weakly_accept',
                     'weakly_reject', 'strongly_reject']].values
    imgs = [strongly_accept_mag, weakly_accept_mag,
            weakly_reject_mag, strongly_reject_mag]
    data = [img.get_data() for img in imgs]

    data_avg = np.average(np.stack(data, -1), axis=3, weights=props)
    aff = strongly_accept_mag.affine  # same for all
    img_avg = nib.Nifti1Image(data_avg, aff)
    img_avg.to_filename(out_file)


def run_first_levels():
    subjects = get_subjects()
    for sub in subjects[:1]:
        smooth_files(sub)
        run_first_level(sub, mod='gain')
        run_first_level(sub, mod='loss')


if __name__ == '__main__':
    run_first_levels()
