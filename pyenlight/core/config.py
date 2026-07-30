import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class PhysicsConfig:
    """Core physical constants and orientation vectors."""
    q: float = 1.60217663e-19       
    kB: float = 1.380649e-23        
    c0: float = 299792458.0         
    hP: float = 6.62607015e-34      
    T0: float = 300.0               
    bK: float = 2.8977729e-3        
    pd_peak: float = 2e9            
    T: float = 298                  
    eo: float = 8.854e-12           
    

   
    # orientation vectors
    zp: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    zm: np.ndarray = field(default_factory=lambda: np.array([0, 0, -1]))
    xp: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0]))
    xm: np.ndarray = field(default_factory=lambda: np.array([-1, 0, 0]))
    yp: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    ym: np.ndarray = field(default_factory=lambda: np.array([0, -1, 0]))

@dataclass
class SpectralConfig:
    """Spectral integration parameters."""
    l_min: float = 200e-9           
    l_max: float = 1300e-9          
    grid_points: int = 1000         
    t_sun: float = 5800             

@dataclass
class EnvironmentConfig:
    """Room and environmental defaults."""
    reflectivity: float = 0.6       
    wall_resolution: Tuple[int, int] = (20, 20) 
    m: int = 1                      
    rx_area: float = 1e-4           
    fov: float = np.pi/2            
    bounces: int = 4                
    room_dim: np.ndarray = field(default_factory=lambda: np.array([5, 5, 3]))    
    ris_element_area: float = 1e-4

@dataclass
class HardwareConfig:
    """MCU and electrical component specs."""
    f_mcu: float = 16e6             
    f_s: float = 1e3                
    voltage: float = 3.3            
    I_mcu: float = 2.73e-3           
    I_adc: float = 0.7e-3           
    I_ext: float = 1.0e-3           
    I_sleep: float = 2e-6        
    I_wake: float = 1e-3          
    I_tia: float = 0.7e-3         
    
    # Battery parameters
    battery_capacity_mAh: float = 500
    initial_soc: float = 1.0          
    V_batt: float = 3.6             
    mpp_eff: float = 0.8            

@dataclass
class CommConfig:
    """Communication and task parameters."""
    IR_Tx_power: float = 15e-3          
    VLC_Tx_power: float = 1.0            
    sensitivity: float = -100           
    uplink_type: int = 0                
    rx_type: int = 0
    
    L_up_bits: int = 1024           
    L_dw_bits: int = 128             
    N_s_up: int = 100               
    N_c_up: float = 1e3             
    bit_rate_up_ir: float = 10e3   
    bit_rate_up_rf: float = 250e3   
    bit_rate_dw: float = 10e3        
    t_init: float = 5e-3            
    t_wait: float = 1e-3           
    T_cycle: float = 60.0          
    harvesting_hours: float = 5.0     
    n_sp: float = 0.4               

@dataclass
class DeviceConfig:
    """Dictionary-based parameters for specific hardware models."""
    # RF Channel defaults
    rf_channel: Dict = field(default_factory=lambda: {
        'n': 1.46, 'pl_ref': 34.62, 'k': 2.03, 'f': 2.45, 
        'sigma': 3.76, 'sigma_factor': 2
    })
    # TIA Model defaults
    tia: Dict = field(default_factory=lambda: {
        'RF': 1e6, 'Vn': 15e-9, 'In': 400e-15,
        'fncV': 1e3, 'fncI': 1e3, 'temperature': 300.0
    })
    # PV Physics defaults 
    pv_circuit: Dict = field(default_factory=lambda: {
        'A': 1e-4, 'n': 1.6, 'Rs': 1.0, 'Rsh': 1000.0, 'Voc': 0.64, 'Jsc': 35e-3,
        'Lo': 1e-6, 'Co': 1e-6, 'Rc': 10.0, 
        'Na': 1.0e22, 'Nd': 1.0e25, 'L': 300e-6, 'er': 11.68, 'ni': 1.0e16,
        'Gref': 1000.0, 'f_steps': 600
    })
    # IR Driver defaults
    ir_driver: Dict = field(default_factory=lambda: {
        'imax': 100e-3, 'imin': 0.0,
        'pol': np.array([1.353e-01, 1.868e-01, -1.017e-04]),
        'polinv': np.array([-1.740e+01, 5.329e+00, 5.618e-04])
    })
    # RF Driver defaults
    rf_driver: Dict = field(default_factory=lambda: {
        'p_min': -20.0, 'p_max': 5.0, 'pol': np.array([0.24, 8.8])
    })

@dataclass
class EnLightConfig:
    """Master configuration container for EnLight-IoT."""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    devices: DeviceConfig = field(default_factory=DeviceConfig)
