import numpy as np
from enlight_iot.core.config import EnLightConfig
from enlight_iot.core.utils import to_scal_Nx1, Qfunction, Qinv
from enlight_iot.environment.geometry import RoomBuilder, Room
from enlight_iot.environment.channel import ChannelEngine
from enlight_iot.network.nodes import NodeBuilder, SNManager, MNManager, ANManager
from enlight_iot.hardware.devices import PV
from enlight_iot.core.interface import PhyResultsDTO

class oPhyGains:
    """
    Optical Physical Layer Gain Calculator.
    Central physics engine for the link budget. 
    """
    def __init__(self, room, masters: MNManager, sensors: SNManager, ambient: ANManager, config: EnLightConfig, btma = True):
        self.config = config
        self.room = room
        self.mn = masters
        self.sn = sensors
        self.intra_mac_gains = not btma
        
        self.ambient = ambient
        
        # Initialize the refactored Channel Engine
        self.engine = ChannelEngine(self.config, self.room)
        
        self.compute_gains()
        self.compute_ambient()

    def compute_gains(self):
        # ── Downlink ─────────────────────────────────────────────────────────
        self.h_d_los  = self.engine.los_gains(self.mn.OTx_elements, self.sn.ORx_elements)
        self.h_d_diff = self.engine.diffuse_gains(self.mn.OTx_elements, self.sn.ORx_elements, bounces=self.config.env.bounces)
        self.h_d_ris  = self.engine.ris_gains(self.mn.OTx_elements, self.sn.ORx_elements)

        # ── Uplink (IR) ───────────────────────────────────────────────────────
        if self.sn.ir_flag > 0:
            self.h_u_los  = self.engine.los_gains(self.sn.OTx_elements, self.mn.ORx_elements)
            self.h_u_diff = self.engine.diffuse_gains(self.sn.OTx_elements, self.mn.ORx_elements, bounces=self.config.env.bounces)
            self.h_u_ris  = self.engine.ris_gains(self.sn.OTx_elements, self.mn.ORx_elements)

            #intra-sensors gains for cca-use
            if self.intra_mac_gains:
                self.h_ss_los  = self.engine.los_gains(self.sn.OTx_elements, self.sn.ORx_elements)
                self.h_ss_diff = self.engine.diffuse_gains(self.sn.OTx_elements, self.sn.ORx_elements, bounces=self.config.env.bounces)
                self.h_ss_ris  = self.engine.ris_gains(self.sn.OTx_elements, self.sn.ORx_elements)

        # ── Uplink (RF) ───────────────────────────────────────────────────────
        if self.sn.rf_flag > 0:
            self.h_u_rf = self.engine.rf_gains(self.sn.RFTx_elements, self.mn.ORx_elements)
            self.h_ss_rf = self.engine.rf_gains(self.sn.RFTx_elements, self.sn.RFTx_elements)

    def compute_downlink(self):
        self.p_d_los  = self.h_d_los  * self.mn.OTx_elements.p
        self.p_d_diff = self.h_d_diff * self.mn.OTx_elements.p
        self.p_d_ris  = self.h_d_ris  * self.mn.OTx_elements.p

        self.i_d_los  = self.p_d_los  * self.sn.c_d
        self.i_d_diff = self.p_d_diff * self.sn.c_d
        self.i_d_ris  = self.p_d_ris  * self.sn.c_d

        self.i_d_signal = self.i_d_los + self.i_d_diff + self.i_d_ris

    def compute_uplink(self):
        if self.sn.ir_flag > 0:
            self.p_u_los  = self.h_u_los  * self.sn.OTx_elements.p
            self.p_u_diff = self.h_u_diff * self.sn.OTx_elements.p
            self.p_u_ris  = self.h_u_ris  * self.sn.OTx_elements.p

            self.i_u_los  = self.p_u_los  * self.mn.c_d
            self.i_u_diff = self.p_u_diff * self.mn.c_d
            self.i_u_ris  = self.p_u_ris  * self.mn.c_d

            self.i_u_signal = self.i_u_los + self.i_u_diff + self.i_u_ris

            #additional code for the intra-nodes metrics (CCA)
            if self.intra_mac_gains:
                self.p_ss_los  = self.h_ss_los  * self.sn.OTx_elements.p
                self.p_ss_diff = self.h_ss_diff * self.sn.OTx_elements.p
                self.p_ss_ris  = self.h_ss_ris  * self.sn.OTx_elements.p

                self.i_ss_los  = self.p_ss_los  * self.sn.c_ss
                self.i_ss_diff = self.p_ss_diff * self.sn.c_ss
                self.i_ss_ris  = self.p_ss_ris  * self.sn.c_ss

                self.i_ss_signal = self.i_ss_los + self.i_ss_diff + self.i_ss_ris

    def compute_ambient(self):
        self.ix_d_noise = np.zeros((1, self.sn.ORx_elements.N))
        self.is_d_noise = np.zeros((1, self.sn.ORx_elements.N))
        self.ix_u_noise = np.zeros((1, self.mn.ORx_elements.N))
        self.is_u_noise = np.zeros((1, self.mn.ORx_elements.N))

        # ── Artificial ambient (lamps) ────────────────────────────────────────
        if self.ambient is not None:
            # Downlink receivers (at Sensors)
            self.hx_d_los  = self.engine.los_gains(self.ambient.OTx_elements, self.sn.ORx_elements)
            self.hx_d_diff = self.engine.diffuse_gains(self.ambient.OTx_elements, self.sn.ORx_elements, bounces=self.config.env.bounces)
            self.px_d_los  = self.hx_d_los  * self.ambient.OTx_elements.p
            self.px_d_diff = self.hx_d_diff * self.ambient.OTx_elements.p
            self.ix_d_los  = self.px_d_los  * self.sn.c_d
            self.ix_d_diff = self.px_d_diff * self.sn.c_d
            self.ix_d_noise = (self.ix_d_los + self.ix_d_diff).reshape(-1, self.sn.no_sensors)

            # Uplink receivers (at Masters)
            self.hx_u_los  = self.engine.los_gains(self.ambient.OTx_elements, self.mn.ORx_elements)
            self.hx_u_diff = self.engine.diffuse_gains(self.ambient.OTx_elements, self.mn.ORx_elements, bounces=self.config.env.bounces)
            self.px_u_los  = self.hx_u_los  * self.ambient.OTx_elements.p
            self.px_u_diff = self.hx_u_diff * self.ambient.OTx_elements.p
            self.ix_u_los  = self.px_u_los  * self.mn.c_d
            self.ix_u_diff = self.px_u_diff * self.mn.c_d
            self.ix_u_noise = (self.ix_u_los + self.ix_u_diff).reshape(-1, self.mn.no_masters)

        # ── Natural ambient (sunlight through windows) ────────────────────────
        if self.room.Tx_windows_elements is not None:
            # Uplink receivers (at Masters)
            self.hs_u_los  = self.engine.los_gains(self.room.Tx_windows_elements, self.mn.ORx_elements)
            self.hs_u_diff = self.engine.diffuse_gains(self.room.Tx_windows_elements, self.mn.ORx_elements, bounces=self.config.env.bounces)
            self.ps_u_los  = self.hs_u_los  * self.room.Tx_windows_elements.p
            self.ps_u_diff = self.hs_u_diff * self.room.Tx_windows_elements.p
            self.is_u_los  = self.ps_u_los  * self.mn.c_d_n
            self.is_u_diff = self.ps_u_diff * self.mn.c_d_n
            self.is_u_noise = (
                np.sum(self.is_u_los,  axis=0) +
                np.sum(self.is_u_diff, axis=0)
            ).reshape(-1, self.mn.no_masters)            

            # Downlink receivers (at Sensors)
            self.hs_d_los  = self.engine.los_gains(self.room.Tx_windows_elements, self.sn.ORx_elements)
            self.hs_d_diff = self.engine.diffuse_gains(self.room.Tx_windows_elements, self.sn.ORx_elements, bounces=self.config.env.bounces)
            self.ps_d_los  = self.hs_d_los  * self.room.Tx_windows_elements.p
            self.ps_d_diff = self.hs_d_diff * self.room.Tx_windows_elements.p
            self.is_d_los  = self.ps_d_los  * self.sn.c_d_n
            self.is_d_diff = self.ps_d_diff * self.sn.c_d_n
            
            self.is_d_noise = (
                np.sum(self.is_d_los,  axis=0) +
                np.sum(self.is_d_diff, axis=0)
            ).reshape(-1, self.sn.no_sensors)
            
 

class PhyNet:
    """
    Physics Network Simulation Kernel.
    """
    def __init__(self, design, budget_run=False, **kwargs):
        self.config = kwargs.get('config', EnLightConfig())
        self.room_builder = RoomBuilder(design, self.config)
        self.room = Room(self.room_builder, self.config)
        self.sn = NodeBuilder(design, "sensors", self.config)
        self.mn = NodeBuilder(design, "masters", self.config)
        self.btma = kwargs.get('btma_mode', True)
        self.intra_mac_gains = not self.btma
        
        if "ambient_nodes" in design.get("nodes", {}):
            self.an = NodeBuilder(design, "ambient_nodes", self.config)
        else:
            self.an = None
       
        self.snm = SNManager(self.sn, self.config)
        self.mnm = MNManager(self.mn, self.config)
        self.amn = ANManager(self.an) if self.an is not None else None
        
        self.ogains = oPhyGains(self.room, self.mnm, self.snm, self.amn, self.config, btma = self.btma)  
        
        self.pv_kwargs = design.get("PV_circuit", {})  

        self.compute_noise()
        if budget_run == True:
            self.set_tx_power()
        self.ogains.compute_downlink()
        self.ogains.compute_uplink()
        self.compute_metrics()

    def calc_min_ow_tx_power(self, target_ber):
        self.target_ber = to_scal_Nx1(self.snm.ir_flag, target_ber)
        self.target_g = Qinv(self.target_ber)
        self.target_snr = self.target_g**2
        
        # Safely ignore division by zero specifically for this math block
        with np.errstate(divide='ignore', invalid='ignore'):
            self.p_req_los = 2 * self.target_g * np.sqrt(self.x_u_noise) / (self.ogains.h_u_los * self.mnm.c_d)
            self.p_req_diff = 2 * self.target_g * np.sqrt(self.x_u_noise) / (self.ogains.h_u_diff * self.mnm.c_d)
            self.p_req_ris = 2 * self.target_g * np.sqrt(self.x_u_noise) / (self.ogains.h_u_ris * self.mnm.c_d)
            
            h_total = self.ogains.h_u_ris + self.ogains.h_u_los + self.ogains.h_u_diff
            self.p_req_total = 2 * self.target_g * np.sqrt(self.x_u_noise) / (h_total * self.mnm.c_d)

        self.p_req = np.min(self.p_req_total, axis=1).reshape(-1, 1) 
        self.p_sel = np.argmin(self.p_req_total, axis=1, keepdims=True)

    def calc_min_rf_tx_power(self):
        self.p_rf_x = self.mnm.sensitivity.T + self.ogains.h_u_rf
        self.p_rf = np.min(self.p_rf_x, axis=1).reshape(-1, 1) 
        self.p_rf_sel = np.argmin(self.p_rf_x, axis=1, keepdims=True) 

    def set_tx_power(self, target_ber=3.8e-3):
        if self.snm.rf_flag > 0:
            self.calc_min_rf_tx_power()
            self.snm.RFTx_elements.p = self.p_rf
            
        if self.snm.ir_flag > 0:  
            self.align_sensors_to_master()
            self.ogains.compute_gains()
            self.calc_min_ow_tx_power(target_ber)
            self.snm.OTx_elements.p = self.p_req

    def compute_noise(self):
        self.g_d_noise = None
        self.flag_pv = (self.snm.ORx_elements.type_Rx == 1).flatten()
        self.no_pv = np.sum(self.flag_pv)
        if self.flag_pv.any():
            self.g_d_noise = np.zeros((1, self.no_pv))
            if self.amn is not None:  
                self.gix_d_noise = self.ogains.ix_d_noise[:, self.flag_pv] / self.snm.ORx_elements.A.flatten()[self.flag_pv]
                self.g_d_noise = self.gix_d_noise
            if self.room.Tx_windows_elements is not None:
                self.gis_d_noise = self.ogains.is_d_noise[:, self.flag_pv] / self.snm.ORx_elements.A.flatten()[self.flag_pv]
                self.g_d_noise = self.g_d_noise + self.gis_d_noise
        
        self.Rb_d = to_scal_Nx1(self.snm.no_sensors, self.mnm.Rb_down)
        self.n_sp_d = to_scal_Nx1(self.snm.no_sensors, self.mnm.n_sp_d)    
        self.BW_d = self.Rb_d / self.n_sp_d

        if self.intra_mac_gains:
            self.BW_ss = np.zeros(self.snm.no_sensors)
            ir_mask = (self.sn.uplink_type == 0).flatten()
        
            self.BW_ss[ir_mask] = self.snm.Rb_up_ir.flatten() / self.snm.n_sp_u.flatten()
            self.BW_ss[~ir_mask] = self.BW_d.flatten()[~ir_mask]
          
        self.x_d_noise = None
        self.flag_pd = (self.snm.ORx_elements.type_Rx == 0).flatten()
        if self.flag_pd.any():
            self.tia_noise_downlink = self.snm.tia.calc_noise_power(self.BW_d.reshape(-1,))[self.flag_pd]
            self.x_d_noise = self.snm.tia.calc_noise_power(self.BW_d.reshape(-1,))[self.flag_pd]
            if self.room.Tx_windows_elements is not None:
                self.xis_d_noise = 2 * self.config.physics.q * self.ogains.is_d_noise[:, self.flag_pd] * self.BW_d.reshape(-1,)[self.flag_pd]
                self.x_d_noise = self.tia_noise_downlink + self.xis_d_noise
            if self.amn is not None:
                self.xix_d_noise = 2 * self.config.physics.q * self.ogains.ix_d_noise[:, self.flag_pd] * self.BW_d.reshape(-1,)[self.flag_pd]
                self.x_d_noise = self.x_d_noise + self.xix_d_noise.sum(axis = 0)

            if self.intra_mac_gains:
                self.tia_noise_ss = self.snm.tia.calc_noise_power(self.BW_ss)[self.flag_pd]
                self.x_ss_noise = self.tia_noise_ss.copy()
            
            if self.room.Tx_windows_elements is not None and self.intra_mac_gains:
                self.xis_ss_noise = 2 * self.config.physics.q * self.ogains.is_d_noise[:, self.flag_pd] * self.BW_ss[self.flag_pd]
                self.x_ss_noise += self.xis_ss_noise.sum(axis = 0)   # ← flatten (1,N) → (N,)
            if self.amn is not None and self.intra_mac_gains:
                self.xix_ss_noise = 2 * self.config.physics.q * self.ogains.ix_d_noise[:, self.flag_pd] * self.BW_ss[self.flag_pd]
                self.x_ss_noise += self.xix_ss_noise.sum(axis = 0)   # ← same fix

        self.x_u_noise = None
        self.Rb_u = self.snm.Rb_up.reshape(-1, 1)
        if self.snm.ir_flag > 0:
            self.Rb_u_ir = to_scal_Nx1(self.snm.ir_flag, self.snm.Rb_up_ir)        
            self.n_sp_u = to_scal_Nx1(self.snm.ir_flag, self.snm.n_sp_u)   
            self.BW_u = self.Rb_u_ir / self.n_sp_u  
            self.tia_noise_uplink = self.mnm.tia.calc_noise_power(self.BW_u.reshape(-1,))
            self.x_u_noise = self.mnm.tia.calc_noise_power(self.BW_u.reshape(-1,)).reshape(-1, 1)
            if self.room.Tx_windows_elements is not None:
                self.xis_u_noise = 2 * self.config.physics.q * self.ogains.is_u_noise * self.BW_u.reshape(-1, 1)
                self.x_u_noise = self.tia_noise_uplink.reshape(-1, 1) + self.xis_u_noise
            if self.amn is not None:
                self.xix_u_noise = 2 * self.config.physics.q * np.sum(self.ogains.ix_u_noise, axis=0) * self.BW_u.reshape(-1, 1)
                self.x_u_noise = self.x_u_noise + self.xix_u_noise

    def align_sensors_to_master(self):
        if self.snm.ir_flag == 0:
            print("No Optical Uplinks to align.")
            return

        m_pos = self.mnm.ORx_elements.r
        if m_pos.shape[0] > 1:
            print("Multiple master nodes, not implemented yet")
            return

        s_pos = self.snm.OTx_elements.r
        direction = m_pos - s_pos
        
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        new_nT = np.divide(direction, norms, out=np.zeros_like(direction), where=norms!=0)

        self.snm.OTx_elements.n = new_nT.reshape(-1, 3)
        print("Sensors aligned...")

    def compute_metrics(self):
        CHUNK_SIZE = 50

        self.snr_d = np.zeros(self.snm.no_sensors)
        self.snr_d_dB = np.zeros(self.snm.no_sensors)

        self.snr_ss = np.zeros((self.snm.no_sensors, self.snm.no_sensors))
        self.snr_ss_dB = np.zeros((self.snm.no_sensors, self.snm.no_sensors))

        self.snr_d_los_dB = np.full(self.snm.no_sensors, -1000.0)
        self.snr_d_diff_dB = np.full(self.snm.no_sensors, -1000.0)
        self.snr_d_ris_dB = np.full(self.snm.no_sensors, -1000.0)
        self.snr_d_diff_tot_dB = np.full(self.snm.no_sensors, -1000.0)

        # Identify which rows correspond to the IR transmitters
        ir_mask = (self.sn.uplink_type == 0).flatten()

        if self.flag_pv.any():
            self.g_d_los = self.ogains.i_d_los[:, self.flag_pv] / self.snm.ORx_elements.A.flatten()[self.flag_pv]
            self.g_d_diff = self.ogains.i_d_diff[:, self.flag_pv] / self.snm.ORx_elements.A.flatten()[self.flag_pv]
            self.g_d_ris = self.ogains.i_d_ris[:, self.flag_pv] / self.snm.ORx_elements.A.flatten()[self.flag_pv]

            self.g_d_signal = self.g_d_los + self.g_d_diff + self.g_d_ris
            self.g_d_signal_total = np.max(self.g_d_signal, axis=0)

            _Gsig_dl = self.g_d_signal_total.flatten()
            _Gamb_dl = np.maximum(self.g_d_noise.flatten(), 1e-6)
            _A_dl = self.snm.ORx_elements.A.flatten()[self.flag_pv]
            _total_dl = len(_Gsig_dl)

            _signal_pv_flat = np.zeros(_total_dl)
            _noise_pv_flat = np.zeros(_total_dl)
            _ind_dl = np.zeros(_total_dl, dtype=int)
            _Pmax_dl = np.zeros((_total_dl, 1))
            _BW_dl = np.zeros((_total_dl, 1))
            _V_dl = None
            _I_dl = None

            for _start in range(0, _total_dl, CHUNK_SIZE):
                _end = min(_start + CHUNK_SIZE, _total_dl)
                _chunk = PV(
                    self.config,
                    Gsignal=_Gsig_dl[_start:_end],
                    Gamb=_Gamb_dl[_start:_end],
                    A=_A_dl[_start:_end],
                    unscaled=True, run=True, **self.pv_kwargs
                )
                _ind_dl[_start:_end] = _chunk.ind.flatten()
                _signal_pv_flat[_start:_end] = np.take_along_axis(
                    _chunk.vac[..., -1], _chunk.ind.reshape(-1, 1), axis=1
                ).flatten() / 0.707
                _noise_pv_flat[_start:_end] = 4 * (
                    np.take_along_axis(_chunk.th_noise, _chunk.ind.reshape(-1, 1), axis=1).flatten() +
                    np.take_along_axis(_chunk.sh_noise, _chunk.ind.reshape(-1, 1), axis=1).flatten()
                )
                _Pmax_dl[_start:_end] = _chunk.Pmax
                _BW_dl[_start:_end] = _chunk.bw_ind
                if _V_dl is None:
                    _V_dl = _chunk.V
                    _I_dl = _chunk.I
                else:
                    _V_dl = np.vstack([_V_dl, _chunk.V])
                    _I_dl = np.vstack([_I_dl, _chunk.I])
                del _chunk

            class _PVProxy:
                pass
            self.pvx = _PVProxy()
            self.pvx.ind = _ind_dl.reshape(-1, 1)
            self.pvx.V = _V_dl
            self.pvx.I = _I_dl
            self.pvx.Pmax = _Pmax_dl
            self.pvx.BW = _BW_dl

            self.signal_pv = _signal_pv_flat.reshape(-1, 1)
            self.noise_pv = _noise_pv_flat.reshape(-1, 1)
            self.snr_pv = self.signal_pv**2 / self.noise_pv
            self.snr_pv_dB = 10 * np.log10(self.snr_pv)

            self.snr_d[self.flag_pv] = self.snr_pv.reshape(-1,)
            self.snr_d_dB[self.flag_pv] = self.snr_pv_dB.reshape(-1,)
            
            if self.snm.ir_flag > 0 and self.intra_mac_gains:
            	self.g_ss_los = self.ogains.i_ss_los[:, self.flag_pv] / self.snm.ORx_elements.A.flatten()[self.flag_pv]
            	self.g_ss_diff = self.ogains.i_ss_diff[:, self.flag_pv] / self.snm.ORx_elements.A.flatten()[self.flag_pv]
            	self.g_ss_ris = self.ogains.i_ss_ris[:, self.flag_pv] / self.snm.ORx_elements.A.flatten()[self.flag_pv]

            	self.g_ss_signal = self.g_ss_los + self.g_ss_diff + self.g_ss_ris

            	num_tx = self.g_ss_signal.shape[0]
            	num_pv = self.g_ss_signal.shape[1]

            	g_ss_signal_flat = self.g_ss_signal.flatten()
            	g_d_noise_tiled = np.maximum(np.tile(self.g_d_noise, (num_tx, 1)).flatten(), 1e-6)
            	A_tiled = np.tile(self.snm.ORx_elements.A.flatten()[self.flag_pv], (num_tx, 1)).flatten()

            	_total_ss = len(g_ss_signal_flat)
            	signal_pv_ss_flat = np.zeros(_total_ss)
            	noise_pv_ss_flat = np.zeros(_total_ss)

            	for _start in range(0, _total_ss, CHUNK_SIZE):
                	_end = min(_start + CHUNK_SIZE, _total_ss)
                	_chunk = PV(
                    	self.config,
                    	Gsignal=g_ss_signal_flat[_start:_end],
                    	Gamb=g_d_noise_tiled[_start:_end],
                    	A=A_tiled[_start:_end],
                    	unscaled=True, run=True, **self.pv_kwargs
                	)
                	signal_pv_ss_flat[_start:_end] = np.take_along_axis(
                    	_chunk.vac[..., -1], _chunk.ind.reshape(-1, 1), axis=1
                	).flatten() / 0.707
                	noise_pv_ss_flat[_start:_end] = 4 * (
                    	np.take_along_axis(_chunk.th_noise, _chunk.ind.reshape(-1, 1), axis=1).flatten() +
                    	np.take_along_axis(_chunk.sh_noise, _chunk.ind.reshape(-1, 1), axis=1).flatten()
                	)
                	del _chunk

            	self.snr_pv_ss = (signal_pv_ss_flat**2 / noise_pv_ss_flat).reshape(num_tx, num_pv)

            	with np.errstate(divide='ignore'):
                	self.snr_pv_dB_ss = 10 * np.log10(self.snr_pv_ss)

            	# Map the rows to the nodes sending IR signals
            	self.snr_ss[np.ix_(ir_mask, self.flag_pv)] = self.snr_pv_ss
            	self.snr_ss_dB[np.ix_(ir_mask, self.flag_pv)] = self.snr_pv_dB_ss

        if self.flag_pd.any():

            self.x_d_los = self.ogains.i_d_los[:, self.flag_pd]
            self.x_d_diff = self.ogains.i_d_diff[:, self.flag_pd]
            self.x_d_ris = self.ogains.i_d_ris[:, self.flag_pd]

            self.x_d_signal = self.x_d_los + self.x_d_diff + self.x_d_ris
            self.x_d_signal_total = np.sum(self.x_d_signal, axis=0)

            self.snr_pd = self.x_d_signal_total**2 / (4 * self.x_d_noise)
            self.snr_pd_dB = 10 * np.log10(self.snr_pd)

            self.snr_d[self.flag_pd] = self.snr_pd.reshape(-1,)
            self.snr_d_dB[self.flag_pd] = self.snr_pd_dB.reshape(-1,)

            i_los_total = np.sum(self.x_d_los, axis=0)
            i_diff_total = np.sum(self.x_d_diff, axis=0)
            i_ris_total = np.sum(self.x_d_ris, axis=0)

            self.snr_d_los_dB[self.flag_pd] = 10 * np.log10(np.maximum(i_los_total**2 / (4 * self.x_d_noise), 1e-60)).flatten()
            self.snr_d_diff_dB[self.flag_pd] = 10 * np.log10(np.maximum(i_diff_total**2 / (4 * self.x_d_noise), 1e-60)).flatten()
            self.snr_d_ris_dB[self.flag_pd] = 10 * np.log10(np.maximum(i_ris_total**2 / (4 * self.x_d_noise), 1e-60)).flatten()
            self.snr_d_diff_tot_dB[self.flag_pd] = 10 * np.log10(np.maximum((i_ris_total+i_diff_total)**2 / (4 * self.x_d_noise), 1e-60)).flatten()
            

            if self.snm.ir_flag > 0 and self.intra_mac_gains:
                self.x_ss_los = self.ogains.i_ss_los[:, self.flag_pd]
                self.x_ss_diff = self.ogains.i_ss_diff[:, self.flag_pd]
                self.x_ss_ris = self.ogains.i_ss_ris[:, self.flag_pd]
                
                self.x_ss_signal = self.x_ss_los + self.x_ss_diff + self.x_ss_ris
                
                self.snr_pd_ss = self.x_ss_signal**2 / (4 * self.x_ss_noise)
                self.snr_pd_dB_ss = 10 * np.log10(self.snr_pd_ss)
                
                self.snr_ss[np.ix_(ir_mask, self.flag_pd)] = self.snr_pd_ss
                self.snr_ss_dB[np.ix_(ir_mask, self.flag_pd)] = self.snr_pd_dB_ss
                
                self.snr_ss_dB[np.isneginf(self.snr_ss_dB)] = -1000
                self.BER_ss = Qfunction(np.sqrt(self.snr_ss))
                
                

        self.snr_d_dB[np.isneginf(self.snr_d_dB)] = -1000
        

        self.BER_d = Qfunction(np.sqrt(self.snr_d))
        

        if self.snm.rf_flag > 0:
            self.hrf = self.ogains.h_u_rf
            self.p_u_rf_m = self.snm.RFTx_elements.p - self.hrf
            self.rf_margin = self.p_u_rf_m - self.mnm.sensitivity.T
            self.rf_best_margin = np.max(self.rf_margin, axis=1)
            self.u_sel_rf = np.argmax(self.rf_margin, axis=1, keepdims=True)
            self.p_u_rf = np.take_along_axis(self.p_u_rf_m, self.u_sel_rf, axis=1)
            self.phy_pdr_up_rf = (self.rf_best_margin >= 0).astype(float)
            
            self.p_ss_rf_rx = self.snm.RFTx_elements.p - self.ogains.h_ss_rf
            self.hidden_node_mask_rf = self.p_ss_rf_rx < self.snm.sensitivity.T 
            np.fill_diagonal(self.hidden_node_mask_rf, False)

        if self.snm.ir_flag > 0:
            self.snr_u = self.ogains.i_u_signal**2 / (4 * self.x_u_noise)
            self.snr_u_dB = 10 * np.log10(self.snr_u)
            self.BER_u = Qfunction(np.sqrt(self.snr_u))
            self.snr_u_sel = np.max(self.snr_u, axis=1)
            self.u_sel_ow = np.argmax(self.snr_u, axis=1, keepdims=True)
            self.ber_u_sel = np.take_along_axis(self.BER_u, self.u_sel_ow, axis=1)

            self.snr_u_los_dB = 10 * np.log10(np.maximum(self.ogains.i_u_los**2 / (4 * self.x_u_noise), 1e-20))
            self.snr_u_diff_dB = 10 * np.log10(np.maximum(self.ogains.i_u_diff**2 / (4 * self.x_u_noise), 1e-20))
            self.snr_u_ris_dB = 10 * np.log10(np.maximum(self.ogains.i_u_ris**2 / (4 * self.x_u_noise), 1e-20))
            self.snr_u_diff_tot_dB = 10 * np.log10(np.maximum((self.ogains.i_u_diff+self.ogains.i_u_ris)**2 / (4 * self.x_u_noise), 1e-20))
            
            self.snr_u_los_sel = np.max(self.snr_u_los_dB, axis=1)
            self.snr_u_diff_sel = np.max(self.snr_u_diff_dB, axis=1)
            self.snr_u_ris_sel = np.max(self.snr_u_ris_dB, axis=1)
            self.snr_u_diff_tot_sel = np.max(self.snr_u_diff_tot_dB, axis=1)
        
        self.g_d_los_total = np.zeros(self.snm.no_sensors)
        self.g_d_diff_total = np.zeros(self.snm.no_sensors)
        self.g_d_ris_total = np.zeros(self.snm.no_sensors)

        if self.flag_pv.any():
            self.g_d_los_total[self.flag_pv] = np.sum(self.g_d_los, axis=0)
            self.g_d_diff_total[self.flag_pv] = np.sum(self.g_d_diff, axis=0)
            self.g_d_ris_total[self.flag_pv] = np.sum(self.g_d_ris, axis=0)


    def save_phy_state(self, filepath="phy_matrices.npz"):
        """Saves heavy physical layer matrices and isolated link components."""
        
        # 1. Base Physical Metrics (always calculated)
        save_dict = {
            'positions_sensors': self.snm.ORx_elements.r,
            'snr_downlink_total_dB': getattr(self, 'snr_d_dB', None),
            'snr_uplink_total_dB': getattr(self, 'snr_u_dB', None),
            'snr_intra_sensor_dB': getattr(self, 'snr_ss_dB', None),
            'ber_downlink': getattr(self, 'BER_d', None),
            'ber_uplink': getattr(self, 'BER_u', None),
        }

        # 2. Add PD Downlink isolated SNRs
        if self.flag_pd.any():
            save_dict['snr_d_pd_los_dB'] = getattr(self, 'snr_d_los_dB', None)
            save_dict['snr_d_pd_diff_dB'] = getattr(self, 'snr_d_diff_dB', None)
            save_dict['snr_d_pd_diff_tot_dB'] = getattr(self, 'snr_d_diff_tot_dB', None)
            save_dict['snr_d_pd_ris_dB'] = getattr(self, 'snr_d_ris_dB', None)

        # 3. Add PV Downlink isolated Optical Gains
        if self.flag_pv.any():
            save_dict['g_d_pv_los_opt_gain'] = getattr(self, 'g_d_los_total', None)
            save_dict['g_d_pv_diff_opt_gain'] = getattr(self, 'g_d_diff_total', None)
            save_dict['g_d_pv_ris_opt_gain'] = getattr(self, 'g_d_ris_total', None)

        # 4. Add OW Uplink isolated SNRs
        if self.snm.ir_flag > 0:
            save_dict['snr_u_ow_los_dB'] = getattr(self, 'snr_u_los_sel', None)
            save_dict['snr_u_ow_diff_dB'] = getattr(self, 'snr_u_diff_sel', None)
            save_dict['snr_u_ow_ris_dB'] = getattr(self, 'snr_u_ris_sel', None)
            save_dict['snr_u_ow_diff_tot__sel_dB'] = getattr(self, 'snr_u_diff_tot_sel', None)
            
        # 5. Clean out any 'None' values just in case
        save_dict = {k: v for k, v in save_dict.items() if v is not None}

        # Save to compressed numpy format
        np.savez_compressed(filepath, **save_dict)

    def export_energy_telemetry(self):
        
        has_pv = hasattr(self, 'pvx')
        pv_v = np.take_along_axis(self.pvx.V, self.pvx.ind.reshape(-1, 1), axis=1).flatten() if has_pv else np.array([])
        pv_i = np.take_along_axis(self.pvx.I, self.pvx.ind.reshape(-1, 1), axis=1).flatten() if has_pv else np.array([])
        
        otx_p = self.snm.OTx_elements.p.reshape(-1,) if hasattr(self.snm, 'OTx_elements') else np.array([])
        rftx_p = self.snm.RFTx_elements.p.reshape(-1,) if hasattr(self.snm, 'RFTx_elements') else np.array([])

        return PhyResultsDTO(
            no_sensors=int(self.snm.no_sensors),
            rb_up=getattr(self, 'Rb_u', np.array([])).flatten(),
            rb_down=getattr(self, 'Rb_d', np.array([])).flatten(),
            flag_pv=getattr(self, 'flag_pv', np.zeros(self.snm.no_sensors, dtype=bool)),
            uplink_type=self.sn.uplink_type,
            otx_p=otx_p,
            rftx_p=rftx_p,
            snr_d_dB=getattr(self, 'snr_d_dB', np.full(self.snm.no_sensors, -1000.0)),
            snr_ss_dB=getattr(self, 'snr_ss_dB', np.full((self.snm.no_sensors, self.snm.no_sensors), -1000.0)),
            snr_u_dB=getattr(self, 'snr_u_dB', np.full((self.snm.no_sensors, 1), -1000.0)),
            phy_pdr_up_rf=getattr(self, 'phy_pdr_up_rf', np.zeros(self.snm.no_sensors)),
            hidden_node_mask_rf=getattr(self, 'hidden_node_mask_rf', np.zeros((self.snm.no_sensors, self.snm.no_sensors), dtype=bool)),
            pv_v_active=pv_v,
            pv_i_active=pv_i
        )

