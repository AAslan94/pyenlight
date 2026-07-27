import numpy as np
from enlight_iot.core.config import EnLightConfig
from enlight_iot.core.utils import as_array_of_size
from enlight_iot.hardware.devices import IRdriver, RF_calc_I
from enlight_iot.network.mac import call_MAC

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

    def __init__(self, phy_net, design, config: EnLightConfig, MAC=False, btma_mode=False):
        self.config    = config
        self.MAC       = MAC
        self.btma_mode = btma_mode
        self.pn        = phy_net
        self.N         = self.pn.snm.no_sensors

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
        self.Rb_up   = self.pn.Rb_u.flatten()
        self.Rb_down = self.pn.Rb_d.flatten()
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
        if hasattr(self.pn, 'flag_pv') and np.any(self.pn.flag_pv):
            try:
                self.harvesting_hours[self.pn.flag_pv] = hh_input
            except ValueError:
                print("Warning: 'harvesting_hours' size mismatch with PV nodes. Using 0.")

        # 6. MAC Simulation Configuration 
        # Checks for a 'MAC' dictionary in energy_profile, falling back to 'protocol'
        mac_cfg = self.u_prof.get('MAC', self.u_prot)
        self.mac_sim_time = mac_cfg.get('sim_time_us', 30e8)
        self.mac_n_seeds  = mac_cfg.get('n_seeds', 150)
        self.mac_snr_th   = mac_cfg.get('SNR_THRESHOLD_dB', 8.5)
        self.mac_bt_th    = mac_cfg.get('BUSY_TONE_THRESHOLD_dB', 8.5)
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

        # Pull the already-broadcasted array from NodeBuilder!
        self.ir_m = (self.pn.sn.uplink_type == 0).flatten()
        self.rf_m = (self.pn.sn.uplink_type == 1).flatten()

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
        self.I_rx   = self.I_mcu + self.I_adc + self.I_tia

        self.I_tx = np.zeros(self.N)
        if np.any(self.ir_m):
            self.I_tx[self.ir_m] = (
                self.I_mcu[self.ir_m]
                + 0.5 * self.ir_driver.calc_I(self.pn.snm.OTx_elements.p.reshape(-1,))
            )
        if np.any(self.rf_m):
            # Fixed: Passing self.config
            self.I_tx[self.rf_m] = RF_calc_I(
                self.pn.snm.RFTx_elements.p.reshape(-1,), self.config, **self.rf_config
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

        self.d_total = (
            self.d_init + self.d_sens_u + self.d_proc_u
            + self.d_tx + self.d_cca + self.d_wait + self.d_rx
        )
        self.E_sleep = self.V * self.I_sleep * np.maximum(0, self.T_cycle - self.d_total)
        self.E_cycle = self.E_active + self.E_sleep

    # ──────────────────────────────────────────────────────────────────────────
    # MAC integration
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_mac_times(self):
        """
        Replace IR-node duration arrays with MAC-simulation-derived per-node
        phase times. RF nodes are never modified.
        """
        # ── Run the MAC simulation if not already cached ──────────────────────
        if not hasattr(self, 'MAC_result'):
            SNR_THRESHOLD_dB = 8.5
            ir_mask          = (self.pn.sn.uplink_type == 0).flatten()
            snr_ss_ir_dB     = self.pn.snr_ss_dB[np.ix_(ir_mask, ir_mask)]
            hidden_node_mask = snr_ss_ir_dB < SNR_THRESHOLD_dB
            np.fill_diagonal(hidden_node_mask, False)

            # --- NEW: PHY-Layer Binary Thresholds ---
            snr_up_dB   = np.max(self.pn.snr_u_dB, axis=1).flatten() 
            snr_down_dB = self.pn.snr_d_dB[ir_mask]
            
            phy_pdr_up   = (snr_up_dB >= SNR_THRESHOLD_dB).astype(float)
            phy_pdr_down = (snr_down_dB >= SNR_THRESHOLD_dB).astype(float)
            
            n_ir_nodes = int(ir_mask.sum())

            bt_hidden_mask = None
            if self.btma_mode:
                BUSY_TONE_THRESHOLD_dB = 8.5
                snr_d_ir_dB    = self.pn.snr_d_dB[ir_mask]
                bt_hidden_mask = snr_d_ir_dB < BUSY_TONE_THRESHOLD_dB

            print(f"Running MAC simulation for {n_ir_nodes} IR nodes...")
            self.MAC_result = call_MAC(
                n_ir_nodes,
                float(self.T_cycle[0]),
                sim_time_us       = self.mac_sim_time,
                data_rate_bps     = float(self.Rb_up[self.ir_m][0]),
                symbol_rate_sym_s = float(self.Rb_up[self.ir_m][0]),
                payload_bytes     = int(self.L_up[self.ir_m][0] / 8),
                n_seeds           = self.mac_n_seeds,
                phy_pdr_up        = phy_pdr_up,    # Pass to MAC
                phy_pdr_down      = phy_pdr_down,  # Pass to MAC
                hidden_node_mask  = None if self.btma_mode else hidden_node_mask,
                bt_hidden_mask    = bt_hidden_mask,
                btma_mode         = self.btma_mode,
                log               = self.mac_log,
                debug             = self.mac_debug,
            )

        # ── Initialise per-node metric arrays ─────────────────────────────────
        self._mac_pdr            = np.ones(self.N)
        self._mac_collision_rate = np.zeros(self.N)

        per_node   = self.MAC_result.get('per_node', None)
        xr_m = (self.pn.sn.uplink_type == 0).flatten()
        ir_indices = np.where(xr_m)[0]    # sensor-array indices of IR nodes

        # ── NEW: Print Unreliable Node Warnings ──
        for idx, node_id in enumerate(ir_indices):
            if phy_pdr_up[idx] == 0.0 or phy_pdr_down[idx] == 0.0:
                print(f"Warning: Node {node_id} cannot send / receive packets from the AP reliably in this config (SNR < {SNR_THRESHOLD_dB} dB).")

        if per_node is not None and len(per_node) == len(ir_indices):
            # ── Per-node path ─────────────────────────────────────────────────
            for mac_idx, sensor_idx in enumerate(ir_indices):
                nd = per_node[mac_idx]
                self.d_tx  [sensor_idx] = nd['mean_time_tx_us']  * 1e-6
                self.d_rx  [sensor_idx] = nd['mean_time_rx_us']  * 1e-6
                self.d_cca [sensor_idx] = nd['mean_time_cca_us'] * 1e-6
                self.d_wait[sensor_idx] = nd['mean_time_idle_us'] * 1e-6
                self._mac_pdr           [sensor_idx] = nd['pdr']
                self._mac_collision_rate[sensor_idx] = nd['collision_rate']

        else:
            # ── Fallback: global mean (backward compatible) ───────────────────
            mu = self.MAC_result['mean']
            self.d_tx  [self.ir_m] = mu['mean_time_tx_us']  * 1e-6
            self.d_rx  [self.ir_m] = mu['mean_time_rx_us']  * 1e-6
            self.d_cca [self.ir_m] = mu['mean_time_cca_us'] * 1e-6
            self.d_wait[self.ir_m] = mu['mean_time_idle_us'] * 1e-6
            for sensor_idx in ir_indices:
                self._mac_pdr           [sensor_idx] = mu.get('pdr', 1.0)
                self._mac_collision_rate[sensor_idx] = mu.get('collision_rate', 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Harvesting
    # ──────────────────────────────────────────────────────────────────────────

    def calc_harv_energy(self):
        self.p_raw  = np.zeros(self.N)
        self.p_harv = np.zeros(self.N)

        if self.pn.flag_pv.any():
            v = np.take_along_axis(self.pn.pvx.V, self.pn.pvx.ind.reshape(-1, 1), axis=1)
            i = np.take_along_axis(self.pn.pvx.I, self.pn.pvx.ind.reshape(-1, 1), axis=1)
            self.p_raw [self.pn.flag_pv] = (v * i).flatten()
            self.p_harv[self.pn.flag_pv] = (v * i * self.mpp_eff).flatten()
        else:
            pass # Removed print to match original flow quietly

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
        if has_mac:
            hdr += f"{'PDR':<7} {'ColRate':<9}"
        print(hdr)
        print("-" * len(hdr))

        for i in range(self.N):
            node_type = "PV" if self.pn.flag_pv[i] else "PD"
            link_type = "IR" if self.pn.sn.uplink_type[i] == 0 else "RF"
            life_str  = "Inf" if self.days_to_empty[i] == np.inf else f"{self.days_to_empty[i]:.1f}"

            row = (f"{i:<6} {node_type:<5} {link_type:<4} "
                   f"{self.E_day_consumed[i]:<13.3f} {self.E_day_harvested[i]:<13.3f} "
                   f"{self.E_day_net[i]:<13.3f} {life_str:<11}")
            if has_mac:
                row += f"{self._mac_pdr[i]:<7.3f} {self._mac_collision_rate[i]:<9.4f}"
            print(row)
