"""
Miscellaneous utility functions
"""
import re
import os.path as op
from glob import glob

import numpy as np
import nibabel as nib


def get_run(file_):
    """
    Get run substring from filename
    """
    return re.findall('(run-[0-9]+)_', file_)[0]


def get_subjects():
    """
    Get full list of subjects with fMRIPrep data
    """
    in_dir = '/scratch/kbott/narps/'
    subjects = sorted([op.basename(op.splitext(f)[0]) for f in
                       glob(op.join(in_dir, 'derivatives/fmriprep/*.html'))])
    return subjects


def calhoun_correction(can, temp, disp=None):
    """
    Computation of magnitude beta images from canonical HRF, temporal
    derivative, and dispersion derivative images. From
    https://www.sciencedirect.com/science/article/pii/S105381190300781X
    """
    use_img = False
    if isinstance(can, nib.Nifti1Image):
        use_img = True
        aff = can.affine
        can = can.get_data()
        temp = temp.get_data()
        if disp is not None:
            disp = disp.get_data()

    if disp is not None:
        out = np.sign(can) * np.sqrt((can ** 2) + (temp ** 2) + (disp ** 2))
    else:
        out = np.sign(can) * np.sqrt((can ** 2) + (temp ** 2))

    if use_img:
        out = nib.Nifti1Image(out, aff)

    return out
