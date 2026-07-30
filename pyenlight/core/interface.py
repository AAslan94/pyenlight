import numpy as np
from dataclasses import dataclass

@dataclass
class PhyResultsDTO:
    # 1. Init Setup
    no_sensors: int
    rb_up: np.ndarray
    rb_down: np.ndarray
    flag_pv: np.ndarray
    
    # 2. Cycle Energy
    uplink_type: np.ndarray
    otx_p: np.ndarray
    rftx_p: np.ndarray
    
    # 3. MAC Simulation Arrays
    snr_d_dB: np.ndarray
    snr_ss_dB: np.ndarray
    snr_u_dB: np.ndarray  
    phy_pdr_up_rf: np.ndarray
    hidden_node_mask_rf: np.ndarray
    
    # 4. Harvesting
    pv_v_active: np.ndarray
    pv_i_active: np.ndarray

    def save_npz(self, filepath: str):
        """Saves telemetry to a compressed .npz file."""
        np.savez_compressed(filepath, **self.__dict__)

    @classmethod
    def load_npz(cls, filepath: str) -> 'PhyResultsDTO':
        """Loads telemetry and prevents 0-d array bugs."""
        with np.load(filepath, allow_pickle=True) as data:
            kwargs = {key: data[key] for key in data.files}
            if 'no_sensors' in kwargs:
                kwargs['no_sensors'] = int(kwargs['no_sensors'].item())
            return cls(**kwargs)
