"""Image reconstruction using pc-mri."""
import datetime
from contextlib import redirect_stdout
from math import pi

import numpy as np

from flow.data.neonatalCND import recon


class ZeroFilledRecon(recon.ZeroFilledRecon):
    def __call__(self, kspace, nt=24, device=-1, **kwargs):
        """
        Args:
            kspace: K-space data from scanner.

        Returns:
            img: Reconstructed image, shape (nt, nx, nx).
        """
        log = self.set_up_outs()

        # Puts everything in this context for redirecting prints
        with redirect_stdout(log):
            print('\n\n---------- START ----------')
            print(f'Date Time: {datetime.datetime.now()}')

            img = self.run(kspace, nt, device, **kwargs)

            print(f'Date Time: {datetime.datetime.now()}')
            print('---------- END ----------')

        self.clean_up(log)
        return img

    def set_up_outs(self):
        # File to output log
        logfile = str('/tmp/share/dependency/log.txt')
        log = open(logfile, 'a')
        return log

    def run(self, kspace, nt, device, save_psf=False, **kwargs):
        data = self.load_data(kspace)
        traj, angles = self.get_trajectory(data['kspace'])
        mps = recon.get_csm(kspace, traj, device=device)
        kspace_gated, traj_gated, _ = self.gate(data['kspace'], traj,
                                                angles, data['triggers'],
                                                nt, data['tr'],
                                                data['times'][0])

        # Some memory management
        data['kspace'] = None  # Releases memory if no more references
        data['kspace_gated'] = kspace_gated
        kspace_gated = None
        data['traj_gated'] = traj_gated
        traj_gated = None
        # Reconstruct image
        img = self.recon(data, mps, device, **kwargs)
        img = self.crop(img)
        img_mag, img_phase = phase_difference(img[0], img[1])
        img = img_mag * np.exp(img_phase*1j)
        return img

    def load_data(self, kspace):
        # Transpose data
        kspace = np.transpose(kspace, (0, 2, 1))

        # Change to dictionary format
        data = {}
        data['kspace'] = kspace
        kspace = None

        # Separate the velocity encodes
        nv = 2
        ncoils, na, ns = data['kspace'].shape
        tmp = np.zeros((ncoils, nv, int(na/nv), ns), dtype=np.complex64)
        for v in range(nv):
            tmp[:, v, :, :] = data['kspace'][:, v::nv, :]
        data['kspace'] = tmp
        tmp = None

        # Create time stamps for each spoke
        times0 = 0
        tr = 14
        data['times'] = np.linspace(times0, times0 + (na-1)*(tr/nv), num=na,
                                    dtype=np.float64)

        # Get the trigger times with pseudo heart rate
        rr_interval = 1000  # ms
        # Assume 100 heart beats
        num_beats = 100
        triggers = np.linspace(times0, times0 + (num_beats-1)*rr_interval,
                               num=num_beats, dtype=np.float64)
        data['triggers'] = triggers

        # Hard-code other params for now
        data['dx'] = 1
        data['dy'] = 1
        data['tr'] = tr
        data['venc'] = 150
        data['weight'] = 80

        print('\n----- Summary of Raw Data -----')
        print(f'K-space Shape: {data["kspace"].shape}')
        print(f'Times Shape: {data["times"].shape}')
        print(f'Triggers Shape: {data["triggers"].shape}')
        print(f'Pixel Widths: ({data["dx"]} x {data["dy"]}) mm')
        print(f'TR: {data["tr"]} ms')
        print(f'VENC: {data["venc"]} cm/s')
        print(f'Patient Weight: {data["weight"]:.2f} kg')
        print('')
        return data


def phasecontrast2d(data):
    zerofilled_recon = ZeroFilledRecon()
    img = zerofilled_recon(data)
    img = np.abs(img)
    # img = np.abs(np.angle(img) + pi)
    return img
