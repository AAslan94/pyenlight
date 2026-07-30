import numpy as np
from typing import Dict
from pyenlight.core.config import EnLightConfig
from pyenlight.core.utils import to_scal_Nx1, as_array_of_size, normalize_bool_array
from pyenlight.environment.geometry import OpticalTxElements, OpticalRxElements, RFTxElements
from pyenlight.hardware.physics import SpectralPhysics
from pyenlight.hardware.devices import TIA

class NodeBuilder():
    """
    Parses design configurations to prepare parameters for node creation.

    This class acts as a factory helper, extracting raw data from the design dictionary
    and normalizing it (e.g., reshaping arrays, applying defaults) so that 
    Master, Sensor, or AmbientNode objects can be instantiated cleanly.
    """
    def __init__(self, design: Dict, node_type: str, config: EnLightConfig, console=False):
        self.design = design
        self.node_type = node_type
        self.config = config
        self.get_node_params(node_type, console)
        self.sanity_check()

    def get_node_params(self, node_type, console=False):
        """
        Extracts params with dynamic fallbacks to EnLightConfig.
        """
        # Retrieve the specific node group from the design
        node_group = self.design.get("nodes", {}).get(node_type, {})
        
        # --- 1. Position & Orientation ---
        # Position is the only strictly required field in practice
        self.positions = node_group.get("positions").reshape(-1, 3)
        self.N_nodes = self.positions.shape[0]

        # --- 2. Geometric & Optical Defaults ---
        # fallbacks now point to EnLightConfig
        self.rx_area = node_group.get("rx_area", self.config.env.rx_area)
        self.m = node_group.get("m", self.config.env.m)
        self.FOV = node_group.get("FOV", self.config.env.fov)

        # --- 3. Type & Uplink Defaults ---
        self.rx_type = as_array_of_size(node_group.get("rx_type", 0), self.N_nodes)
        self.uplink_type = node_group.get("uplink_type", self.config.comm.uplink_type)
        
        # --- 4. Electrical & Transmit Power ---
        self.tx_power = node_group.get("tx_power", self.config.comm.VLC_Tx_power)
        self.IR_tx_power = node_group.get("IR_tx_power", self.config.comm.IR_Tx_power)
        self.RF_tx_power = node_group.get("RF_tx_power", self.config.devices.rf_driver['p_min'])
        
        # --- 5. TIA with fallback ---
        self.tia = self.design.get("TIA", self.config.devices.tia)

        # --- 6. Communication Metrics ---
        energy_prof = self.design.get('energy_profile', {})
        comm_cfg = energy_prof.get('communication', {})
        
        self.n_sp_d = comm_cfg.get("n_sp_d", self.config.comm.n_sp)
        
        # Bit rate fallbacks check specific IR/RF/VLC defaults
        self.Rb_down = comm_cfg.get("Rb_down", self.config.comm.bit_rate_dw)
        
        if node_type == "sensors":
            self.nT = node_group.get("nT", self.config.physics.zp)
            self.nR = node_group.get("nR", self.config.physics.zp)

            self.Rb_up = np.zeros(self.N_nodes)
            self.n_sp_u = np.zeros(self.N_nodes)

            design_rb = comm_cfg.get('Rb_up')
            design_nsp = comm_cfg.get('n_sp_u', self.config.comm.n_sp)
            
            self.sensitivity = node_group.get("sensitivity", self.config.comm.sensitivity)

            ir_mask = (self.uplink_type == 0)
            rf_mask = (self.uplink_type == 1)
            
            if design_rb is not None:
                self.Rb_up = as_array_of_size(design_rb, self.N_nodes)
            else:
                self.Rb_up[ir_mask] = self.config.comm.bit_rate_up_ir
                self.Rb_up[rf_mask] = self.config.comm.bit_rate_up_rf
                
            n_sp_u = as_array_of_size(design_nsp, self.N_nodes)
            self.n_sp_u = n_sp_u[ir_mask]
            self.Rb_up_ir = self.Rb_up[ir_mask] 
            self.VLC_pass_filter = node_group.get("VLC_pass_filter", False)
            self.IR_pass_filter = None
              
        elif node_type == "masters":
            self.nT = node_group.get("nT", self.config.physics.zm)
            self.nR = node_group.get("nR", self.config.physics.zm)
            self.IR_pass_filter = node_group.get("IR_pass_filter", True)
            self.sensitivity = node_group.get("sensitivity", self.config.comm.sensitivity)
            self.VLC_pass_filter = None

        else:
            self.nT = node_group.get("nT", self.config.physics.zm)
            self.nR = node_group.get("nR", self.config.physics.zm)
  
    def sanity_check(self):
        """
        Validates parameter dimensions and filters optical properties for hybrid networks.
        """
        #size sanity check for different input styles for uplinks
        if self.node_type != "sensors":
            pass
        else:
            self.uplink_type = to_scal_Nx1(self.positions.reshape(-1,3).shape[0], self.uplink_type).flatten()
            self.no_optical_uplinks = np.where(np.array([self.uplink_type])==0)[0].size
            self.no_RF_uplinks = np.where(np.array([self.uplink_type])==1)[0].size
            self.nT = np.array(self.nT)
            self.m = np.array(self.m)

            if self.nT.reshape(-1,3).shape[0] != 1 and self.nT.reshape(-1,3).shape[0] != self.no_optical_uplinks:
              if self.nT.reshape(-1,3).shape[0] == self.positions.reshape(-1,3).shape[0]:
                nT_x = self.nT[self.uplink_type == 0]
                self.nT = nT_x

            if self.m.reshape(-1,1).shape[0] != 1 and self.m.reshape(-1,1).shape[0] != self.no_optical_uplinks:
              if self.nT.reshape(-1,1).shape[0] == self.positions.reshape(-1,3).shape[0]:
                m_x = self.m[self.uplink_type == 0]
                self.m = m_x


class SNManager:
    """
    Sensor Node Manager.
    """
    def __init__(self, nb: NodeBuilder, config: EnLightConfig):
        self.config = config
        self.spec_phys = SpectralPhysics(self.config)
        self.tia = TIA(self.config, **nb.tia)
        
        self.no_sensors = nb.positions.shape[0]
        self.Rb_up = nb.Rb_up   #transform shape in PhyNet
        self.n_sp_u = nb.n_sp_u #transform shape in Phynet
        self.Rb_up_ir = nb.Rb_up_ir  
        self.rf_flag = 0
        self.ir_flag = 0
        self.ORx_elements = OpticalRxElements( r = nb.positions, n = nb.nR, type_Rx = nb.rx_type, fov = nb.FOV, A = nb.rx_area)
        
        if nb.no_optical_uplinks > 0:
            self.OTx_elements = OpticalTxElements( r = nb.positions[nb.uplink_type == 0 ], n = nb.nT, m = nb.m, p = nb.IR_tx_power)
            self.ir_flag = nb.no_optical_uplinks
        if nb.no_RF_uplinks > 0:
            self.RFTx_elements = RFTxElements(r = nb.positions[nb.uplink_type == 1], p = to_scal_Nx1(nb.no_RF_uplinks,nb.RF_tx_power))
            self.rf_flag = nb.no_RF_uplinks
            self.sensitivity = to_scal_Nx1(self.rf_flag,nb.sensitivity)
            
            
        #make the masks for the effective responsivity calculations
        self.mask_VLC_filter = normalize_bool_array(nb.VLC_pass_filter,self.no_sensors) & (nb.rx_type.reshape(-1,) == 0)
        self.mask_pd_no_filter = ~normalize_bool_array(nb.VLC_pass_filter,self.no_sensors) & (nb.rx_type.reshape(-1,) == 0)
        self.mask_pv = (nb.rx_type.reshape(-1,) == 1)

        self.c_d = np.zeros([self.no_sensors]) #Array to hold effective responsivity values
        self.c_d_n = np.zeros([self.no_sensors]) #Array to hold effective responsivity values
        self.c_ss = np.zeros([self.no_sensors]) #Array to hold effective responsivity values
        
        self.downlink_effective_responsivity()

    def downlink_effective_responsivity(self):
        """
        Computes the Effective Responsivity ($R_{eff}$) via the spectral overlap integral.
        """
        self.c_d[self.mask_VLC_filter] = self.spec_phys.get_responsivity_by_name("WLED2PDwF")
        self.c_d[self.mask_pv] = self.spec_phys.get_responsivity_by_name("WLED2PV")/self.spec_phys.get_responsivity_by_name("SUN2PV") #different approach for PVs
        self.c_d[self.mask_pd_no_filter] = self.spec_phys.get_responsivity_by_name("WLED2PD")
        #for noise calculations - sun to pd / pv
        self.c_d_n[self.mask_VLC_filter] = self.spec_phys.get_responsivity_by_name("SUN2PDwFv")
        self.c_d_n[self.mask_pd_no_filter] = self.spec_phys.get_responsivity_by_name("SUN2PD")
        self.c_d_n[self.mask_pv] = 1 # for irradiance the sun's spectrum is already taken into account
        #for cca calculations  
        self.c_ss[self.mask_VLC_filter] = self.spec_phys.get_responsivity_by_name("IR2PDwX")
        self.c_ss[self.mask_pv] = self.spec_phys.get_responsivity_by_name("IR2PV")/self.spec_phys.get_responsivity_by_name("SUN2PV") #different approach for PVs
        self.c_ss[self.mask_pd_no_filter] = self.spec_phys.get_responsivity_by_name("IR2PD")  


class MNManager:
    """
    Master Node Manager (Base Stations / Access Points).
    """
    def __init__(self, nb: NodeBuilder, config: EnLightConfig):
        self.config = config
        self.spec_phys = SpectralPhysics(self.config)
        self.tia = TIA(self.config, **nb.tia)
        self.no_masters = nb.positions.shape[0]
        
        self.Rb_down = nb.Rb_down  #transform shape in Phynet
        self.n_sp_d = nb.n_sp_d  #transform shape in Phynet
         
        self.sensitivity = to_scal_Nx1(self.no_masters,nb.sensitivity)
        #make the masks for the effective responsivity calculations
        self.mask_IR_filter = normalize_bool_array(nb.IR_pass_filter,self.no_masters)
        self.mask_pd_no_filter = ~normalize_bool_array(nb.IR_pass_filter,self.no_masters)

        self.ORx_elements = OpticalRxElements( r = nb.positions, n = nb.nR, type_Rx = 0, fov = nb.FOV, A = nb.rx_area)
        self.OTx_elements = OpticalTxElements( r = nb.positions, n = nb.nT, m = nb.m, p = nb.tx_power)

        self.c_d = np.zeros([self.no_masters]) #Array to hold effective responsivity values
        self.c_d_n = np.zeros([self.no_masters]) #Array to hold effective responsivity values

        self.uplink_effective_responsivity()

    def uplink_effective_responsivity(self):
        """
        Computes the Effective Responsivity ($c_d$) for the Uplink Channel.
        """
        self.c_d[self.mask_IR_filter] = self.spec_phys.get_responsivity_by_name("IR2PDwF")
        self.c_d[self.mask_pd_no_filter] = self.spec_phys.get_responsivity_by_name("IR2PD")

        self.c_d_n[self.mask_IR_filter] = self.spec_phys.get_responsivity_by_name("SUN2PDwFi")
        self.c_d_n[self.mask_pd_no_filter] = self.spec_phys.get_responsivity_by_name("SUN2PD")


class ANManager:
    """
    Ambient Node Manager.
    """
    def __init__(self, nb: NodeBuilder):
        self.OTx_elements = OpticalTxElements( r = nb.positions, n = nb.nT, m = nb.m, p = nb.tx_power)
