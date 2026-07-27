import numpy as np
import pandas as pd
from enlight_iot.core.config import EnLightConfig
from enlight_iot.core.utils import as_array_of_size
from enlight_iot.hardware.devices import IRdriver, RF_calc_I
from enlight_iot.network.mac import call_MAC
from enlight_iot.core.interface import PhyResultsDTO


class EnergyManager:
    """
    Energy Consumption & Harvesting Manager.

    Simulates the power profile of sensor nodes over a defined operation cycle.
    Integrates hardware specs, task loads, and MAC-layer communication overhead
    to compute total energy drain per cycle, and calculates harvesting potential
    for PV-equipped nodes.

    Energy Model (cycle phases):
        1. Initialization  — wake-up and boot
        2. Sensing         — ADC sampling and sensor readout
        3. Processing      — MCU computation
        4. Uplink (Tx)     — IR LED or RF transmission (MAC-aware for IR)
        5. CCA             — channel sensing, only with MAC enabled
        6. Turnaround      — idle wait before downlink
        7. Downlink (Rx)   — ACK / data reception
        8. Sleep           — remainder of T_cycle at I_sleep
    """

    def __init__(self, phy_data: PhyResultsDTO, design, config=None, MAC=False, btma_mode=False, MAC_mode='unslotted', **kwargs):
        self.config = config if config else kwargs.get('config', EnLightConfig())
        self.MAC       = MAC
        self.btma_mode = btma_mode
        self.MAC_mode  = MAC_mode

        self.phy_data = phy_data
        self.N = self.phy_data.no_sensors

        self.nodes  = design['nodes']['sensors']
        self.u_prof = design.get('energy_profile', {})
        self.u_prot = design.get('protocol', {})
        self.u_mpp  = design.get('MPP', {})

        # 0. Drivers (Fixed: Passing self.config)
        self.ir_driver = IRdriver(self.config, **self.u_prof.get('IRDriver', {}))
        self.rf_config = self.u_prof.get('RFDriver', {})

        # 1. Hardware
        self.f_mcu   = self._v('f_mcu',    'hardware')
        self.f_s     = self._v('f_s',      'hardware')
        self.V       = self._v('voltage',  'hardware')
        self.I_mcu   = self._v('I_mcu')
        self.I_adc   = self._v('I_adc')
        self.I_ext   = self._v('I_ext')
        self.I_sleep = self._v('I_sleep')
        self.I_wake  = self._v('I_wake')
        self.I_tia   = self._v('I_tia')

        # 2. Task loads
        self.N_s_up = self._v('N_s_up',    'tasks')
        self.N_c_up = self._v('N_c_up',    'tasks')
        self.L_up   = self._v('L_up_bits', 'tasks')
        self.L_dw   = self._v('L_dw_bits', 'tasks')

        # 3. Communication
        self.Rb_up   = self.phy_data.rb_up
        self.Rb_down = self.phy_data.rb_down
        self.t_init  = self._v('t_init')
        self.t_wait  = self._v('t_wait')
        self.T_cycle = self._v('T_cycle')

        # 4. Battery
        self.batt_capacity_mAh = self._v('battery_capacity_mAh', 'battery')
        self.V_batt            = self._v('V_batt',       'battery')
        self.initial_soc       = self._v('initial_soc',  'battery')
        self.batt_charge       = self.batt_capacity_mAh * self.V_batt * 3.6 * self.initial_soc

        # 5. Harvesting — only PV nodes get non-zero harvesting hours
        self.mpp_eff          = self.u_mpp.get('mpp_eff', getattr(self.config.hardware, 'mpp_eff', 0.8))
        self.harvesting_hours = np.zeros(self.N)
        hh_input = self.u_prot.get('harvesting_hours', getattr(self.config.comm, 'harvesting_hours', 5.0))
        if np.any(self.phy_data.flag_pv):
            try:
                self.harvesting_hours[self.phy_data.flag_pv] = hh_input
            except ValueError:
                print("Warning: 'harvesting_hours' size mismatch with PV nodes. Using 0.")

        # 6. MAC Simulation Configuration 
        # Checks for a 'MAC' dictionary in energy_profile, falling back to 'protocol'
        mac_cfg = self.u_prof.get('MAC', self.u_prot)
        self.mac_sim_time = mac_cfg.get('sim_time_us', 30e8)
        self.mac_n_seeds  = mac_cfg.get('n_seeds', 150)
        self.mac_snr_th   = mac_cfg.get('SNR_THRESHOLD_dB', getattr(self.config.comm, 'SNR_THRESHOLD_dB', 8.5))
        self.mac_bt_th    = mac_cfg.get('BUSY_TONE_THRESHOLD_dB', getattr(self.config.comm, 'BUSY_TONE_THRESHOLD_dB', 8.5))
        self.mac_log      = mac_cfg.get('log', True)
        self.mac_debug    = mac_cfg.get('debug', False)

        # 6. Daily stat placeholders
        self.E_day_consumed  = np.zeros(self.N)
        self.E_day_harvested = np.zeros(self.N)
        self.E_day_net       = np.zeros(self.N)
        self.days_to_empty   = np.zeros(self.N)

        self.calc_cycle_energy()
        self.calc_harv_energy()
        self.calc_battery_life()

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _v(self, key, profile_sub=None):
        """Hierarchical lookup updated for EnLightConfig structure."""
        if profile_sub:
            d = self.u_prof.get(profile_sub, {})
            if key in d:
                return as_array_of_size(d[key], self.N)
        if key in self.u_prof:
            return as_array_of_size(self.u_prof[key], self.N)
        if key in self.u_prot:
            return as_array_of_size(self.u_prot[key], self.N)
            
        # Fallback to EnLightConfig dynamically across sub-dataclasses
        for category in ['hardware', 'comm', 'env', 'physics', 'devices']:
            sub_cfg = getattr(self.config, category, None)
            if sub_cfg and hasattr(sub_cfg, key):
                return as_array_of_size(getattr(sub_cfg, key), self.N)
                
        raise AttributeError(f"Parameter '{key}' not found in design or EnLightConfig.")

    # ──────────────────────────────────────────────────────────────────────────
    # Core energy calculation
    # ──────────────────────────────────────────────────────────────────────────

    def calc_cycle_energy(self):
        """
        Compute total energy per operation cycle.

        Phase durations:
            t_proc = N_cycles / f_clk
            t_tx   = L_bits   / R_bps   (overridden per-node when MAC=True)

        Energy:
            E_active = V · Σ(I_state · t_state)
            E_sleep  = V · I_sleep · max(0, T_cycle − t_active)
        """
        # ── Phase 1: baseline durations ───────────────────────────────────────
        self.d_init   = self.t_init
        self.d_sens_u = self.N_s_up / self.f_s
        self.d_proc_u = self.N_c_up / self.f_mcu

        self.ir_m = (self.phy_data.uplink_type == 0).flatten()
        self.rf_m = (self.phy_data.uplink_type == 1).flatten()

        self.d_tx   = self.L_up / self.Rb_up     # single-frame baseline
        self.d_wait = self.t_wait
        self.d_rx   = self.L_dw / self.Rb_down
        self.d_cca  = np.zeros(self.N)           # non-zero only with MAC

        # ── Phase 2: MAC override (IR nodes only) ─────────────────────────────
        if self.MAC:
            self._apply_mac_times()

        # ── Phase 3: currents ─────────────────────────────────────────────────
        self.I_sens = self.I_adc + self.I_mcu + self.I_ext
        self.I_proc = self.I_mcu
        is_pv = self.phy_data.flag_pv.astype(bool).reshape(-1)

        # RX current:
        # - PD nodes use MCU + ADC + TIA
        # - PV nodes use MCU + ADC only
        self.I_rx = self.I_mcu + self.I_adc
        self.I_rx = self.I_rx.copy()
        self.I_rx[~is_pv] += self.I_tia[~is_pv]

        self.I_tx = np.zeros(self.N)
        if np.any(self.ir_m):
            self.I_tx[self.ir_m] = (
                self.I_mcu[self.ir_m]
                + 0.5 * self.ir_driver.calc_I(self.phy_data.otx_p)
            )
        if np.any(self.rf_m):
            self.I_tx[self.rf_m] = RF_calc_I(
                self.phy_data.rftx_p, self.config, **self.rf_config
            )

        # ── Phase 4: energy integration ───────────────────────────────────────
        self.E_active = self.V * (
              self.I_wake * self.d_init       # wake-up / boot
            + self.I_sens * self.d_sens_u     # sensing
            + self.I_proc * self.d_proc_u     # processing
            + self.I_tx   * self.d_tx         # uplink TX (retransmissions included when MAC on)
            + self.I_rx   * self.d_cca        # CCA / channel sensing
            + self.I_mcu  * self.d_wait       # turnaround idle
            + self.I_rx   * self.d_rx         # downlink RX
        )
        
        print("Init period: current is " + str(self.I_wake[0]*1000) + " mA and timing is " + str(self.d_init[0]*1000)+ " ms")
        print("Sensing period: current is " + str(self.I_sens[0]*1000) + " mA and timing is " + str(self.d_sens_u[0]*1000)+ " ms")
        print("Processing period: current is " + str(self.I_proc[0]*1000) + " mA and timing is " + str(self.d_proc_u[0]*1000)+ " ms")
        print("Tx period: current is " + str(self.I_tx[0]*1000) + " mA and timing is " + str(self.d_tx[0]*1000)+ " ms")
        print("Turnaround period: current is " + str(self.I_mcu[0]*1000) + " mA and timing is " + str(self.d_wait[0]*1000)+ " ms")
        print("Rx min period: current is " + str(self.I_rx[0]*1000) + " mA and timing is " + str(np.min(self.d_rx*1000))+ " ms")
        print("Rx max period: current is " + str(self.I_rx[0]*1000) + " mA and timing is " + str(np.max(self.d_rx*1000))+ " ms")

        self.d_total = (
            self.d_init + self.d_sens_u + self.d_proc_u
            + self.d_tx + self.d_cca + self.d_wait + self.d_rx
        )
        self.E_sleep = self.V * self.I_sleep * np.maximum(0, self.T_cycle - self.d_total)
        self.E_cycle = self.E_active + self.E_sleep

        print("max Sleep period: current is " + str(np.min(self.I_sleep*1000)) + " mA and timing is " + str(np.max(self.T_cycle - self.d_total))+ " ms")
        print("min Sleep period: current is " + str(np.min(self.I_sleep*1000)) + " mA and timing is " + str(np.min(self.T_cycle - self.d_total))+ " ms")
    # ──────────────────────────────────────────────────────────────────────────
    # MAC integration
    # ──────────────────────────────────────────────────────────────────────────

    
    
    
    def _apply_mac_times(self):
        """
        Replace IR and RF node duration arrays with MAC-simulation-derived per-node
        phase times. Maintains independent caching and warning structures for both domains.
        """
        # ── Initialise per-node metric arrays ─────────────────────────────────
        self._mac_pdr            = np.ones(self.N)
        self._mac_pdr_ap         = np.ones(self.N)
        self._mac_true_col       = np.zeros(self.N) # <--- True Airwave Collisions
        self._mac_phy_up         = np.zeros(self.N) # <--- Uplink Loss (PHY)
        self._mac_phy_dw         = np.zeros(self.N) # <--- ACK Lost (PHY)
        self._mac_blockage       = np.zeros(self.N) # <--- CCAs always busy (MAC)
        self._mac_retries        = np.zeros(self.N)

        # ── Run the MAC simulation if not already cached ──────────────────────
        if not hasattr(self, '_mac_sims_cached'):
            self._mac_sims_cached = True  # Set cache flag
            SNR_THRESHOLD_dB = self.mac_snr_th
            global_phy_pdr_down = (self.phy_data.snr_d_dB >= SNR_THRESHOLD_dB).astype(float)
            
            # ==================================================================
            # 1. PROCESS IR NODES
            # ==================================================================
            ir_indices = np.where(self.ir_m)[0]
            if len(ir_indices) > 0:
                snr_ss_ir_dB = self.phy_data.snr_ss_dB[np.ix_(self.ir_m, self.ir_m)]
                hidden_node_mask = snr_ss_ir_dB < SNR_THRESHOLD_dB
                np.fill_diagonal(hidden_node_mask, False)

                # Mapping 3 (Cont): Uplink SNR (Axis=1 maxing preserved)
                snr_up_dB = np.max(self.phy_data.snr_u_dB, axis=1).flatten() 
                
                phy_pdr_down_ir = global_phy_pdr_down[self.ir_m]
                phy_pdr_up_ir   = (snr_up_dB >= SNR_THRESHOLD_dB).astype(float)
                
                n_ir_nodes = int(len(ir_indices))

                bt_hidden_mask = None
                if self.btma_mode:
                    BUSY_TONE_THRESHOLD_dB = self.mac_bt_th
                    # Mapping 3 (Cont): BTMA check
                    bt_hidden_mask = self.phy_data.snr_d_dB[self.ir_m] < BUSY_TONE_THRESHOLD_dB
                    
                mean_rb_up_ir = float(np.mean(self.Rb_up[self.ir_m]))
                mean_payload_bytes_ir = int(np.mean(self.L_up[self.ir_m]) / 8)

                print(f"Running MAC simulation for {n_ir_nodes} IR nodes...")

                self.MAC_result_ir = call_MAC(
                    n_ir_nodes,
                    float(self.T_cycle[0]),
                    sim_time_us       = self.mac_sim_time,
                    data_rate_bps     = mean_rb_up_ir,
                    symbol_rate_sym_s = mean_rb_up_ir,
                    payload_bytes     = mean_payload_bytes_ir,
                    n_seeds           = self.mac_n_seeds,
                    phy_pdr_up        = phy_pdr_up_ir,    # Pass to MAC
                    phy_pdr_down      = phy_pdr_down_ir,  # Pass to MAC
                    hidden_node_mask  = None if self.btma_mode else hidden_node_mask,
                    bt_hidden_mask    = bt_hidden_mask,
                    btma_mode         = self.btma_mode,
                    log               = self.mac_log,
                    debug             = self.mac_debug,
                    mode              = self.MAC_mode 
                )

                # ── Print Unreliable Node Warnings (IR) ──
                for idx, node_id in enumerate(ir_indices):
                    if phy_pdr_up_ir[idx] == 0.0 or phy_pdr_down_ir[idx] == 0.0:
                        print(f"Warning: IR Node {node_id} cannot send / receive packets from the AP reliably in this config (SNR < {SNR_THRESHOLD_dB} dB).")

            # ==================================================================
            # 2. PROCESS RF NODES
            # ==================================================================
            rf_indices = np.where(self.rf_m)[0]
            if len(rf_indices) > 0:
                phy_pdr_up_rf       = self.phy_data.phy_pdr_up_rf
                hidden_node_mask_rf = self.phy_data.hidden_node_mask_rf
                snr_down_dB_rf      = self.phy_data.snr_d_dB[self.rf_m]
                phy_pdr_down_rf     = global_phy_pdr_down[self.rf_m]
                
                n_rf_nodes = int(len(rf_indices))
                
                mean_rb_up_rf = float(np.mean(self.Rb_up[self.rf_m]))
                mean_payload_bytes_rf = int(np.mean(self.L_up[self.rf_m]) / 8)

                print(f"Running MAC simulation for {n_rf_nodes} RF nodes...")
                self.MAC_result_rf = call_MAC(
                    n_rf_nodes,
                    float(self.T_cycle[0]),
                    sim_time_us       = self.mac_sim_time,
                    data_rate_bps     = mean_rb_up_rf,
                    symbol_rate_sym_s = mean_rb_up_rf,
                    payload_bytes     = mean_payload_bytes_rf,
                    n_seeds           = self.mac_n_seeds,
                    phy_pdr_up        = phy_pdr_up_rf,
                    phy_pdr_down      = phy_pdr_down_rf,
                    hidden_node_mask  = hidden_node_mask_rf,
                    bt_hidden_mask    = None,
                    btma_mode         = False,
                    log               = self.mac_log,
                    debug             = self.mac_debug,
                    mode              = self.MAC_mode 
                )
                
                # ── Print Unreliable Node Warnings (RF) ──
                for idx, node_id in enumerate(rf_indices):
                    if phy_pdr_up_rf[idx] == 0.0 or phy_pdr_down_rf[idx] == 0.0:
                        print(f"Warning: RF Node {node_id} cannot send / receive packets from the AP reliably (margin < 0 or SNR < {SNR_THRESHOLD_dB} dB).")


        # ======================================================================
        # ── Map Results Back to Global Arrays ─────────────────────────────────
        # ======================================================================
        
        # ── 1. Map IR Data ──
        ir_indices = np.where(self.ir_m)[0]
        if len(ir_indices) > 0 and hasattr(self, 'MAC_result_ir'):
            per_node_ir = self.MAC_result_ir.get('per_node', None)
            if per_node_ir is not None and len(per_node_ir) == len(ir_indices):
                for mac_idx, sensor_idx in enumerate(ir_indices):
                    nd = per_node_ir[mac_idx]
                    self.d_tx  [sensor_idx] = nd['mean_time_tx_us']  * 1e-6
                    self.d_rx  [sensor_idx] = nd['mean_time_rx_us']  * 1e-6
                    self.d_cca [sensor_idx] = nd['mean_time_cca_us'] * 1e-6
                    self.d_wait[sensor_idx] = nd['mean_time_idle_us'] * 1e-6
                    self._mac_pdr            [sensor_idx] = nd['pdr_mac'] 
                    self._mac_pdr_ap         [sensor_idx] = nd['pdr_ap']  
                    self._mac_true_col       [sensor_idx] = nd.get('true_mac_collision_rate', 0.0)
                    self._mac_phy_up         [sensor_idx] = nd.get('phy_error_rate_up', 0.0)
                    self._mac_phy_dw         [sensor_idx] = nd.get('phy_error_rate_down', 0.0)
                    self._mac_blockage       [sensor_idx] = nd.get('drop_no_access', 0.0)
                    self._mac_retries        [sensor_idx] = nd.get('avg_retries', 0.0)
            else:
                mu = self.MAC_result_ir['mean'] 
                self.d_tx  [self.ir_m] = mu['mean_time_tx_us']  * 1e-6
                self.d_rx  [self.ir_m] = mu['mean_time_rx_us']  * 1e-6
                self.d_cca [self.ir_m] = mu['mean_time_cca_us'] * 1e-6
                self.d_wait[self.ir_m] = mu['mean_time_idle_us'] * 1e-6
                for sensor_idx in ir_indices:
                    self._mac_pdr            [sensor_idx] = mu.get('pdr_mac', 1.0)
                    self._mac_pdr_ap         [sensor_idx] = mu.get('pdr_ap', 1.0)   
                    self._mac_true_col       [sensor_idx] = mu.get('true_mac_collision_rate', 0.0)
                    self._mac_blockage       [sensor_idx] = nd.get('drop_no_access', 0.0)
                    self._mac_phy_up         [sensor_idx] = mu.get('phy_error_rate_up', 0.0)
                    self._mac_phy_dw         [sensor_idx] = mu.get('phy_error_rate_down', 0.0)
                    self._mac_blockage       [sensor_idx] = mu.get('drop_no_access', 0.0)
                    self._mac_retries        [sensor_idx] = mu.get('avg_retries_per_pkt', 0.0)

        # ── 2. Map RF Data ──
        rf_indices = np.where(self.rf_m)[0]
        if len(rf_indices) > 0 and hasattr(self, 'MAC_result_rf'):
            per_node_rf = self.MAC_result_rf.get('per_node', None)
            if per_node_rf is not None and len(per_node_rf) == len(rf_indices):
                for mac_idx, sensor_idx in enumerate(rf_indices):
                    nd = per_node_rf[mac_idx]
                    self.d_tx  [sensor_idx] = nd['mean_time_tx_us']  * 1e-6
                    self.d_rx  [sensor_idx] = nd['mean_time_rx_us']  * 1e-6
                    self.d_cca [sensor_idx] = nd['mean_time_cca_us'] * 1e-6
                    self.d_wait[sensor_idx] = nd['mean_time_idle_us'] * 1e-6
                    self._mac_pdr            [sensor_idx] = nd['pdr_mac']
                    self._mac_pdr_ap         [sensor_idx] = nd['pdr_ap'] 
                    self._mac_true_col       [sensor_idx] = nd.get('true_mac_collision_rate', 0.0)
                    self._mac_phy_up         [sensor_idx] = nd.get('phy_error_rate_up', 0.0)
                    self._mac_phy_dw         [sensor_idx] = nd.get('phy_error_rate_down', 0.0)
                    self._mac_blockage       [sensor_idx] = nd.get('drop_no_access', 0.0)
                    self._mac_retries        [sensor_idx] = nd.get('avg_retries', 0.0)
            else:
                mu = self.MAC_result_rf['mean'] 
                self.d_tx  [self.rf_m] = mu['mean_time_tx_us']  * 1e-6
                self.d_rx  [self.rf_m] = mu['mean_time_rx_us']  * 1e-6
                self.d_cca [self.rf_m] = mu['mean_time_cca_us'] * 1e-6
                self.d_wait[self.rf_m] = mu['mean_time_idle_us'] * 1e-6
                for sensor_idx in rf_indices:
                    self._mac_pdr            [sensor_idx] = mu.get('pdr_mac', 1.0)
                    self._mac_pdr_ap         [sensor_idx] = mu.get('pdr_ap', 1.0)   
                    self._mac_true_col       [sensor_idx] = mu.get('true_mac_collision_rate', 0.0)
                    self._mac_phy_up         [sensor_idx] = mu.get('phy_error_rate_up', 0.0)
                    self._mac_phy_dw         [sensor_idx] = mu.get('phy_error_rate_down', 0.0)
                    self._mac_blockage       [sensor_idx] = mu.get('drop_no_access', 0.0)
                    self._mac_retries        [sensor_idx] = mu.get('avg_retries_per_pkt', 0.0)
    # ──────────────────────────────────────────────────────────────────────────
    # Harvesting
    # ──────────────────────────────────────────────────────────────────────────

    def calc_harv_energy(self):
        self.p_raw  = np.zeros(self.N)
        self.p_harv = np.zeros(self.N)

        if np.any(self.phy_data.flag_pv):
            v = self.phy_data.pv_v_active
            i = self.phy_data.pv_i_active
            self.p_raw [self.phy_data.flag_pv] = (v * i).flatten()
            self.p_harv[self.phy_data.flag_pv] = (v * i * self.mpp_eff).flatten()

    # ──────────────────────────────────────────────────────────────────────────
    # Battery lifetime
    # ──────────────────────────────────────────────────────────────────────────

    def calc_battery_life(self):
        if not hasattr(self, 'E_cycle'):
            self.calc_cycle_energy()

        self.current_energy  = self.batt_charge
        self.cycles_per_hour = 3600 / self.T_cycle
        self.E_day_consumed  = self.E_cycle * self.cycles_per_hour * 24.0

        if not hasattr(self, 'p_harv'):
            self.p_harv = np.zeros(self.N)

        self.E_day_harvested = self.p_harv * self.harvesting_hours * 3600.0
        self.E_day_net       = self.E_day_harvested - self.E_day_consumed

        drain_mask = self.E_day_net < 0
        self.days_to_empty = np.full(self.N, np.inf)
        if np.any(drain_mask):
            self.days_to_empty[drain_mask] = (
                self.current_energy[drain_mask] / np.abs(self.E_day_net[drain_mask])
            )

        # ── Build and print report ────────────────────────────────────────────
        has_mac = self.MAC and hasattr(self, '_mac_pdr')

        hdr = (f"{'Node':<6} {'Rx':<5} {'UL':<4} "
               f"{'Cons/Day(J)':<13} {'Harv/Day(J)':<13} "
               f"{'Net/Day(J)':<13} {'Life(Days)':<11}")
        #if has_mac:
        #    hdr += f"{'PDR_AP':<8} {'PDR_MAC':<8} {'TrueCol':<8} {'PhyUp':<8} {'PhyDw':<8} {'Max_CCAs':<9} {'Retries':<8}"
        #print(hdr)
        #print("-" * len(hdr))

        # for i in range(self.N):
        #    node_type = "PV" if self.phy_data.flag_pv[i] else "PD"
        #    link_type = "IR" if self.phy_data.uplink_type[i] == 0 else "RF"
        #    life_str  = "Inf" if self.days_to_empty[i] == np.inf else f"{self.days_to_empty[i]:.1f}"

        #    row = (f"{i:<6} {node_type:<5} {link_type:<4} "
        #           f"{self.E_day_consumed[i]:<13.3f} {self.E_day_harvested[i]:<13.3f} "
        #           f"{self.E_day_net[i]:<13.3f} {life_str:<11}")
        #    if has_mac:
        #        row += (f"{self._mac_pdr_ap[i]:<8.3f} {self._mac_pdr[i]:<8.3f} "
        #                f"{self._mac_true_col[i]:<8.4f} {self._mac_phy_up[i]:<8.4f} {self._mac_phy_dw[i]:<8.4f} "
        #                f"{self._mac_blockage[i]:<9.4f} {self._mac_retries[i]:<8.2f}")
        #    print(row)
    
    def get_results_df(self):
        """Compiles all node-level metrics into a Pandas DataFrame."""
        if not hasattr(self, 'E_day_consumed'):
            self.calc_battery_life()

        # 1. Calculate boolean reachability arrays
        SNR_TH = getattr(self, 'mac_snr_th', 8.5)
        
        # Can the Node hear the AP? (Downlink SNR >= Threshold)
        can_hear_ap = (self.phy_data.snr_d_dB >= SNR_TH).flatten()
        
        # Can the AP hear the Node? (Depends on IR vs RF)
        ap_can_hear_node = np.zeros(self.N, dtype=bool)
        
        # Evaluate IR Uplink (SNR based)
        if np.any(self.ir_m):
            snr_up_ir_dB = np.max(self.phy_data.snr_u_dB, axis=1).flatten()
            ap_can_hear_node[self.ir_m] = snr_up_ir_dB >= SNR_TH
            
        if np.any(self.rf_m):
            ap_can_hear_node[self.rf_m] = self.phy_data.phy_pdr_up_rf.flatten() > 0

        data = {
            'Node_ID': np.arange(self.N),
            'Node_Type': ['PV' if flag else 'PD' for flag in self.phy_data.flag_pv],
            'Link_Type': ['IR' if t == 0 else 'RF' for t in self.phy_data.uplink_type],
            'E_Consumed_J': self.E_day_consumed,
            'E_Harvested_J': self.E_day_harvested,
            'Net_Energy_J': self.E_day_net,
            'Life_Days': self.days_to_empty,
            'Can_Listen_AP': can_hear_ap,          
            'AP_Can_Listen': ap_can_hear_node      
        }
        
        # 3. Append MAC Data
        if self.MAC and hasattr(self, '_mac_pdr'):
            data.update({
                'PDR_AP': self._mac_pdr_ap,
                'PDR_MAC': self._mac_pdr,
                'True_Col_Rate': self._mac_true_col,
                'Phy_Err_Up': self._mac_phy_up,
                'Phy_Err_Dw': self._mac_phy_dw,
                'Blockage_Drop': self._mac_blockage,
                'Avg_Retries': self._mac_retries
            })
            
        return pd.DataFrame(data)

    def save_csv(self, filepath="simulation_results.csv"):
        """Exports the results to a CSV file."""
        df = self.get_results_df()
        df.to_csv(filepath, index=False)
        print(f"Results successfully saved to {filepath}")

    
