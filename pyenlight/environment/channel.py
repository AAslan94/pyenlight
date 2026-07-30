import numpy as np
from pyenlight.core.config import EnLightConfig
from pyenlight.core.utils import solar_panel_angular_efficiency, calculate_blockage_mask, generate_microgrids_vectorized
from pyenlight.environment.geometry import Room

class ChannelEngine:
    """
    Direct refactor of the original Gains class logic.
    Uses the provided EnLightConfig for all physical constants.
    """
    def __init__(self, config: EnLightConfig, room: Room):
        self.config = config
        self.room = room
        
        # Pre-compute wall-to-wall gains once and cache on the Room object.
        if self.room.h_ww is None:
            self.room.h_ww = self.calc_h(
                self.room.Tx_wall_elements,
                self.room.Rx_wall_elements
            )

    def _get_blocker_params(self):
        if not self.room.blockers or not isinstance(self.room.blockers, dict):
            return None, 0.3, 1.7
            
        bp = self.room.blockers.get('positions', None)
        r_h = self.room.blockers.get('radius', 0.3)
        h_h = self.room.blockers.get('height', 1.7)
        
        if bp is None or len(bp) == 0:
            return None, r_h, h_h
            
        return bp, r_h, h_h

    def calc_h(self, tx, rx):
        """Standard Lambertian emission model (Broadcasting)."""
        D_tx_rx = -(tx.r[:, None, :] - rx.r[None, :, :])
        D_tx_rx_norm = np.linalg.norm(D_tx_rx, axis=2)
        # Add epsilon to prevent division by zero warnings
        D_tx_rx_unit = D_tx_rx / (D_tx_rx_norm[..., None] + 1e-15)

        cos_irr = np.maximum(0, np.sum(D_tx_rx_unit * tx.n[:, None, :], axis=2))
        cos_inc = np.maximum(0, np.sum(-D_tx_rx_unit * rx.n[None, :, :], axis=2))

        inc_angle = np.arccos(np.clip(cos_inc, -1.0, 1.0))
        fov_mask_pd = (inc_angle <= rx.fov.T)
        is_sp = (rx.type_Rx.T == 1)

        h = (rx.A.T * (tx.m + 1) * (cos_irr ** tx.m) * cos_inc / (2 * np.pi * D_tx_rx_norm ** 2))

        h = np.where(~is_sp, h * fov_mask_pd, h)

        if np.any(is_sp):
            sp_factor = solar_panel_angular_efficiency(cos_inc)
            h = np.where(is_sp, h * sp_factor, h)

        return np.nan_to_num(h, nan=0)

    def los_gains(self, tx, rx):
        h_los = self.calc_h(tx, rx)
        bp, r_h, h_h = self._get_blocker_params()
        if bp is not None:
            blocked = calculate_blockage_mask(tx.r, rx.r, bp, r_h, h_h)
            h_los[blocked] = 0.0
        return h_los

    def diffuse_gains(self, tx, rx, bounces=4):
        h_mw = self.calc_h(tx, self.room.Rx_wall_elements)
        h_ws = self.calc_h(self.room.Tx_wall_elements, rx)

        bp, r_h, h_h = self._get_blocker_params()
        if bp is not None:
            k_res = 3
            pts = k_res ** 2
            rx_micro = generate_microgrids_vectorized(self.room.Rx_wall_elements.r, self.room.Rx_wall_elements.n, self.room.Rx_wall_elements.A, k=k_res)
            tx_micro = generate_microgrids_vectorized(self.room.Rx_wall_elements.r, self.room.Rx_wall_elements.n, self.room.Rx_wall_elements.A, k=k_res)

            raw_mw_mask = calculate_blockage_mask(tx.r, rx_micro.reshape(-1, 3), bp, r_h, h_h)
            soft_mw_mask = np.mean(raw_mw_mask.reshape(tx.N, self.room.Rx_wall_elements.N, pts), axis=2)
            h_mw *= (1.0 - soft_mw_mask)

            raw_ws_mask = calculate_blockage_mask(tx_micro.reshape(-1, 3), rx.r, bp, r_h, h_h)
            soft_ws_mask = np.mean(raw_ws_mask.reshape(self.room.Tx_wall_elements.N, pts, rx.N), axis=1)
            h_ws *= (1.0 - soft_ws_mask)

        R = np.diag(self.room.Rx_wall_elements.refl.flatten())
        current_wall_power = h_mw @ R
        H_diffuse_total = np.zeros((tx.N, rx.N))

        for k in range(1, bounces + 1):
            H_diffuse_total += current_wall_power @ h_ws
            if k < bounces:
                current_wall_power = (current_wall_power @ self.room.h_ww) @ R

        return H_diffuse_total

    def ris_gains(self, tx, rx):
        if self.room.Tx_RIS_elements is None:
            return np.zeros((tx.N, rx.N))

        r_master = tx.r
        r_sensor = rx.r
        r_ris = self.room.Tx_RIS_elements.r
        n_ris = self.room.Tx_RIS_elements.n

        D_tx_ris = -(r_master[:, None, :] - r_ris[None, :, :])
        d_tx_ris = np.linalg.norm(D_tx_ris, axis=2)
        D_tx_ris_unit = D_tx_ris / (d_tx_ris[..., None] + 1e-15)

        cos_phi = np.maximum(0, np.sum(D_tx_ris_unit * tx.n[:, None, :], axis=2))
        cos_th_in = np.maximum(0, np.sum(-D_tx_ris_unit * n_ris[None, :, :], axis=2))

        D_ris_rx = -(r_ris[:, None, :] - r_sensor[None, :, :])
        d_ris_rx = np.linalg.norm(D_ris_rx, axis=2)
        D_ris_rx_unit = D_ris_rx / (d_ris_rx[..., None] + 1e-15)

        cos_th_out = np.maximum(0, np.sum(D_ris_rx_unit * n_ris[:, None, :], axis=2))
        cos_psi = np.maximum(0, np.sum(-D_ris_rx_unit * rx.n[None, :, :], axis=2))

        A_ris = self.room.Rx_RIS_elements.A.flatten()
        rho_ris = self.room.Rx_RIS_elements.refl.flatten()
        A_rx = rx.A.flatten()

        leg1 = ((tx.m.flatten()[:, None] + 1) / (2 * np.pi) * cos_phi ** tx.m.flatten()[:, None] * cos_th_in / d_tx_ris ** 2)
        ris_term = rho_ris * A_ris
        is_sp = (rx.type_Rx.flatten() == 1)
        fov_mask = (np.arccos(np.clip(cos_psi, -1.0, 1.0)) <= rx.fov.flatten()[None, :])
        leg2 = A_rx[None, :] * cos_th_out * cos_psi / d_ris_rx ** 2
        leg2 = np.where(~is_sp[None, :], leg2 * fov_mask, leg2)

        h_per_elem = (leg1[:, :, None] * ris_term[None, :, None] * leg2[None, :, :])

        if np.any(is_sp):
            sp_factor = solar_panel_angular_efficiency(cos_psi)
            h_per_elem = np.where(is_sp[None, None, :], h_per_elem * sp_factor[None, :, :], h_per_elem)

        bp, r_h, h_h = self._get_blocker_params()
        if bp is not None:
            blocked_leg1 = calculate_blockage_mask(r_master, r_ris, bp, r_h, h_h)
            blocked_leg2 = calculate_blockage_mask(r_ris, r_sensor, bp, r_h, h_h)
            joint_blocked = blocked_leg1[:, :, None] | blocked_leg2[None, :, :]
            h_per_elem = np.where(joint_blocked, 0.0, h_per_elem)

        return np.nan_to_num(np.sum(h_per_elem, axis=1), nan=0)

    def rf_gains(self, tx, rx):
        cfg = self.config.devices.rf_channel
        D_tx_rx = -(tx.r[:, None, :] - rx.r[None, :, :])
        d = np.linalg.norm(D_tx_rx, axis=2)
        return (10 * cfg['n'] * np.log10(d + 1e-15)) + cfg['pl_ref'] + (10 * cfg['k'] * np.log10(cfg['f'])) + (cfg['sigma_factor'] * cfg['sigma'])
