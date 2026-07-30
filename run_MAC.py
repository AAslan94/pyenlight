import copy
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyenlight import EnLightConfig, EnergyManager
from pyenlight.core.interface import PhyResultsDTO
from pyenlight.network import mac as mac_module
from design_A1 import master_design_example
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
})
# =============================================================================
# Experiment settings
# =============================================================================

# Use a small sweep first to verify the blockage/collision logic.
# Set QUICK_TEST = False for the full paper sweep.
QUICK_TEST = False
NODE_SWEEP = [20, 50, 100] if QUICK_TEST else [50, 100, 150, 200, 300, 350, 400, 450, 500]

SECONDS_PER_DAY = 86400.0

HIGH_SNR_DB = 100.0
LOW_SNR_DB = 0.0

T_CYCLE_S = 60.0
RB_UP_BPS = 10e3
RB_DOWN_BPS = 10e3
IR_TX_POWER_W = 15e-3

L_UP_BITS = 1024

ACK_BYTES = 16
L_DW_BITS = ACK_BYTES * 8

MAC_MODE = "unslotted"
MAC_SIM_TIME_US = 3e8 if QUICK_TEST else 30e8
MAC_N_SEEDS = 10 if QUICK_TEST else 150

# Use max_workers=1 for Spyder/Jupyter stability.
# Set to None to use ProcessPoolExecutor parallelism.
MAC_MAX_WORKERS = 6

SNR_THRESHOLD_DB = 8.5
BUSY_TONE_THRESHOLD_DB = 8.5

BLOCKED_FRACTION = 0.10

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots/MAC")


# =============================================================================
# Design + synthetic PHY
# =============================================================================

def prepare_design(base_design, n):
    design = copy.deepcopy(base_design)

    design["nodes"]["sensors"]["positions"] = np.zeros((n, 3))
    design["nodes"]["sensors"]["rx_type"] = 0
    design["nodes"]["sensors"]["uplink_type"] = 0
    design["nodes"]["sensors"]["IR_tx_power"] = IR_TX_POWER_W

    design.setdefault("energy_profile", {})
    design["energy_profile"]["T_cycle"] = T_CYCLE_S

    design["energy_profile"].setdefault("communication", {})
    design["energy_profile"]["communication"]["Rb_up"] = RB_UP_BPS
    design["energy_profile"]["communication"]["Rb_down"] = RB_DOWN_BPS
    design["energy_profile"]["communication"]["n_sp_u"] = 0.4
    design["energy_profile"]["communication"]["n_sp_d"] = 0.4

    design["energy_profile"].setdefault("tasks", {})
    design["energy_profile"]["tasks"]["L_up_bits"] = L_UP_BITS
    design["energy_profile"]["tasks"]["L_dw_bits"] = L_DW_BITS

    design["energy_profile"]["MAC"] = {
        "sim_time_us": MAC_SIM_TIME_US,
        "n_seeds": MAC_N_SEEDS,
        "SNR_THRESHOLD_dB": SNR_THRESHOLD_DB,
        "BUSY_TONE_THRESHOLD_dB": BUSY_TONE_THRESHOLD_DB,
        "log": False,
        "debug": False,
    }

    # No CF is set here. The TIA class computes CF from bandwidth internally.
    design["TIA"] = {
        "RF": 1e6,
        "Vn": 15e-9,
        "In": 400e-15,
        "fncV": 1e3,
        "fncI": 1e3,
        "temperature": 300.0,
    }

    return design


def make_synthetic_dto(n, blocked_fraction=0.0, blockage_mode="none"):
    """
    Build a synthetic PHY data set for controlled MAC tests.

    blockage_mode:
        "none"     : all UL and DL links are above threshold.
        "both"     : selected nodes have both UL and DL below threshold.
        "downlink" : selected nodes have DL below threshold but UL above threshold.

    In BTMA mode, DL-blocked nodes cannot detect the busy tone. The distinction
    between "both" and "downlink" therefore tests the receiver-side collision
    fix:
        - both blocked: the node may transmit, but its UL does not reach the AP
          and must not collide with a valid packet;
        - downlink only: the node may transmit and its UL reaches the AP, so it
          can cause a collision.
    """
    valid_modes = {"none", "both", "downlink"}
    if blockage_mode not in valid_modes:
        raise ValueError(
            f"blockage_mode must be one of {sorted(valid_modes)}, "
            f"got {blockage_mode!r}"
        )

    n_blocked = int(round(blocked_fraction * n))
    blocked_idx = np.arange(n_blocked)

    snr_d = np.full(n, HIGH_SNR_DB)
    snr_u = np.full((n, 1), HIGH_SNR_DB)
    snr_ss = np.full((n, n), HIGH_SNR_DB)
    np.fill_diagonal(snr_ss, HIGH_SNR_DB)

    if blockage_mode in {"both", "downlink"} and n_blocked > 0:
        snr_d[blocked_idx] = LOW_SNR_DB

    if blockage_mode == "both" and n_blocked > 0:
        snr_u[blocked_idx, 0] = LOW_SNR_DB

    return PhyResultsDTO(
        no_sensors=n,

        rb_up=np.full(n, RB_UP_BPS),
        rb_down=np.full(n, RB_DOWN_BPS),
        flag_pv=np.zeros(n, dtype=bool),

        uplink_type=np.zeros(n, dtype=int),
        otx_p=np.full(n, IR_TX_POWER_W),
        rftx_p=np.full(n, -20.0),

        snr_d_dB=snr_d,
        snr_ss_dB=snr_ss,
        snr_u_dB=snr_u,

        phy_pdr_up_rf=np.ones(0),
        hidden_node_mask_rf=np.zeros((0, 0), dtype=bool),

        pv_v_active=np.zeros(n),
        pv_i_active=np.zeros(n),
    )


# =============================================================================
# EnergyManager helpers
# =============================================================================

def make_energy_manager_nomac(dto, design, config):
    with contextlib.redirect_stdout(io.StringIO()):
        em = EnergyManager(
            dto,
            design,
            config=config,
            MAC=False,
            btma_mode=False,
            MAC_mode=MAC_MODE,
        )
    return em


def recompute_energy_after_mac_override(em):
    em.E_active = em.V * (
        em.I_wake * em.d_init
        + em.I_sens * em.d_sens_u
        + em.I_proc * em.d_proc_u
        + em.I_tx * em.d_tx
        + em.I_rx * em.d_cca
        + em.I_mcu * em.d_wait
        + em.I_rx * em.d_rx
    )

    em.d_total = (
        em.d_init
        + em.d_sens_u
        + em.d_proc_u
        + em.d_tx
        + em.d_cca
        + em.d_wait
        + em.d_rx
    )

    em.E_sleep = em.V * em.I_sleep * np.maximum(0, em.T_cycle - em.d_total)
    em.E_cycle = em.E_active + em.E_sleep

    em.E_day_consumed = em.E_cycle * (SECONDS_PER_DAY / em.T_cycle)
    em.E_day_harvested = np.zeros(em.N)
    em.E_day_net = em.E_day_harvested - em.E_day_consumed

    em.days_to_empty = np.where(
        em.E_day_consumed > 0,
        em.batt_charge / em.E_day_consumed,
        np.inf,
    )


def apply_mac_to_energy_manager(em, dto, btma_mode):
    n = dto.no_sensors

    snr_ss_ir_dB = dto.snr_ss_dB
    hidden_node_mask = snr_ss_ir_dB < SNR_THRESHOLD_DB
    np.fill_diagonal(hidden_node_mask, False)

    snr_up_dB = np.max(dto.snr_u_dB, axis=1).flatten()

    phy_pdr_up = (snr_up_dB >= SNR_THRESHOLD_DB).astype(float)
    phy_pdr_down = (dto.snr_d_dB >= SNR_THRESHOLD_DB).astype(float)

    bt_hidden_mask = None
    if btma_mode:
        bt_hidden_mask = dto.snr_d_dB < BUSY_TONE_THRESHOLD_DB

    payload_bytes = int(L_UP_BITS / 8)

    mac_result = mac_module.call_MAC(
        nodes=n,
        period=T_CYCLE_S,
        mode=MAC_MODE,
        traffic_type="periodic",
        n_seeds=MAC_N_SEEDS,
        sim_time_us=MAC_SIM_TIME_US,
        data_rate_bps=RB_UP_BPS,
        symbol_rate_sym_s=RB_UP_BPS,
        payload_bytes=payload_bytes,
        ack_bytes=ACK_BYTES,
        phy_pdr_up=phy_pdr_up,
        phy_pdr_down=phy_pdr_down,
        hidden_node_mask=None if btma_mode else hidden_node_mask,
        bt_hidden_mask=bt_hidden_mask,
        btma_mode=btma_mode,
        log=False,
        debug=False,
        max_workers=MAC_MAX_WORKERS,
    )

    per_node = mac_result["per_node"]

    for i, nd in enumerate(per_node):
        em.d_tx[i] = nd["mean_time_tx_us"] * 1e-6
        em.d_rx[i] = nd["mean_time_rx_us"] * 1e-6
        em.d_cca[i] = nd["mean_time_cca_us"] * 1e-6
        em.d_wait[i] = nd["mean_time_idle_us"] * 1e-6

    recompute_energy_after_mac_override(em)

    return mac_result


# =============================================================================
# Results extraction
# =============================================================================

def mac_mean_metrics(mac_result):
    empty = {
        "PDR_AP": np.nan,
        "PDR_MAC": np.nan,
        "True_MAC_Collision_Rate": np.nan,
        "PHY_UL_Failure_Rate": np.nan,
        "PHY_DL_Failure_Rate": np.nan,
        "Payload_Throughput_kbps": np.nan,
        "Union_Active_Payload_Throughput_Normalized": np.nan,
        "Union_Active_Time_s": np.nan,
        "CCA_Attempts_Per_Generated_Packet": np.nan,
        "Backoff_Slots_Per_Generated_Packet": np.nan,
        "MAC_Blocked_Packets_Per_Generated_Packet": np.nan,
        "Retransmissions_Per_Generated_Packet": np.nan,
    }

    if mac_result is None:
        return empty

    mu = mac_result["mean"]
    params = mac_result["params"]
    per_node = mac_result["per_node"]

    mean_true_collision_rate = float(np.mean([
        nd.get("true_mac_collision_rate", np.nan) for nd in per_node
    ]))
    mean_phy_up_rate = float(np.mean([
        nd.get("phy_error_rate_up", np.nan) for nd in per_node
    ]))
    mean_phy_down_rate = float(np.mean([
        nd.get("phy_error_rate_down", np.nan) for nd in per_node
    ]))

    pkts_gen = max(mu.get("pkts_gen", 0.0), 1.0)
    
    active_payload_kbps = mu.get("active_payload_throughput_kbps", np.nan)
    data_rate_bps = float(getattr(params, "data_rate_bps", RB_UP_BPS))

    active_payload_norm = (
        active_payload_kbps * 1e3 / data_rate_bps
        if data_rate_bps > 0 and not np.isnan(active_payload_kbps)
        else np.nan
    )

    drop_no_access_per_generated_packet = mu.get("drop_no_access", 0.0) / pkts_gen

    return {
        "PDR_AP": mu.get("pdr_ap", np.nan),
        "PDR_MAC": mu.get("pdr_mac", np.nan),
        "True_MAC_Collision_Rate": mean_true_collision_rate,
        "PHY_UL_Failure_Rate": mean_phy_up_rate,
        "PHY_DL_Failure_Rate": mean_phy_down_rate,
        "Payload_Throughput_kbps": mu.get("payload_throughput_kbps", np.nan),
        
        # Dimensionless active-time payload efficiency.
        # Example: 0.76 means ACKed payload throughput during active intervals is 76% of Rb_up.
        "Union_Active_Payload_Throughput_Normalized": active_payload_norm,
        "Union_Active_Time_s": mu.get("active_time_s", np.nan),

        "CCA_Attempts_Per_Generated_Packet": mu.get("cca_per_period", np.nan),
        "Backoff_Slots_Per_Generated_Packet": mu.get("bo_slots_per_period", np.nan),
        "MAC_Blocked_Packets_Per_Generated_Packet": drop_no_access_per_generated_packet,
        "Retransmissions_Per_Generated_Packet": mu.get("avg_retries_per_pkt", np.nan),
    }


def make_node_dataframe(em, dto, case_name, n, blockage_mode, mac_result=None):
    ul_reachable = np.max(dto.snr_u_dB, axis=1).flatten() >= SNR_THRESHOLD_DB
    dl_reachable = dto.snr_d_dB.flatten() >= SNR_THRESHOLD_DB

    df = pd.DataFrame({
        "node_id": np.arange(em.N),
        "case": case_name,
        "N": n,
        "blockage_mode": blockage_mode,
        "UL_reaches_AP": ul_reachable,
        "DL_reaches_node": dl_reachable,
        "E_cycle_J": em.E_cycle,
        "E_day_consumed_J": em.E_day_consumed,
        "Life_days": em.days_to_empty,
        "d_tx_s": em.d_tx,
        "d_rx_s": em.d_rx,
        "d_cca_s": em.d_cca,
        "d_wait_s": em.d_wait,
        "d_total_s": em.d_total,
    })

    if mac_result is not None:
        per_node = mac_result["per_node"]

        for key in [
            "pdr_ap",
            "pdr_mac",
            "true_mac_collision_rate",
            "phy_error_rate_up",
            "phy_error_rate_down",
            "drop_no_access",
            "mean_backoff_slots",
            "mean_cca_attempts",
        ]:
            df[key] = [nd.get(key, np.nan) for nd in per_node]

    return df


def summarize_case(
    em, case_name, n, blockage_mode, blocked_fraction, mac_result=None
):
    row = {
        "case": case_name,
        "N": n,
        "blockage_mode": blockage_mode,
        "blocked_fraction": blocked_fraction,

        "E_cycle_J_mean": float(np.mean(em.E_cycle)),
        "E_cycle_J_std": float(np.std(em.E_cycle)),

        "E_day_J_mean": float(np.mean(em.E_day_consumed)),
        "E_day_J_total": float(np.sum(em.E_day_consumed)),
        "E_day_J_std": float(np.std(em.E_day_consumed)),
        "E_day_J_min": float(np.min(em.E_day_consumed)),
        "E_day_J_max": float(np.max(em.E_day_consumed)),

        "Life_days_mean": float(np.mean(em.days_to_empty)),

        "d_tx_s_mean": float(np.mean(em.d_tx)),
        "d_rx_s_mean": float(np.mean(em.d_rx)),
        "d_cca_s_mean": float(np.mean(em.d_cca)),
        "d_wait_s_mean": float(np.mean(em.d_wait)),
        "d_total_s_mean": float(np.mean(em.d_total)),
    }

    row.update(mac_mean_metrics(mac_result))

    pdr_mac = row["PDR_MAC"]
    if np.isnan(pdr_mac):
        pdr_mac = 1.0

    packets_per_node_per_day = SECONDS_PER_DAY / T_CYCLE_S
    generated_packets_per_day = n * packets_per_node_per_day
    delivered_packets_per_day = generated_packets_per_day * pdr_mac

    row["Generated_Packets_Per_Day"] = float(generated_packets_per_day)
    row["Delivered_Packets_Per_Day"] = float(delivered_packets_per_day)

    row["Energy_Per_Generated_Packet_J"] = (
        row["E_day_J_total"] / generated_packets_per_day
        if generated_packets_per_day > 0
        else np.inf
    )

    row["Energy_Per_Delivered_Packet_J"] = (
        row["E_day_J_total"] / delivered_packets_per_day
        if delivered_packets_per_day > 0
        else np.inf
    )

    return row


def run_case(
    n,
    case_name,
    mac_enabled,
    btma_mode,
    blocked_fraction,
    blockage_mode,
):
    config = EnLightConfig()
    design = prepare_design(master_design_example, n)
    dto = make_synthetic_dto(
        n,
        blocked_fraction=blocked_fraction,
        blockage_mode=blockage_mode,
    )

    em = make_energy_manager_nomac(dto, design, config)

    mac_result = None
    if mac_enabled:
        mac_result = apply_mac_to_energy_manager(em, dto, btma_mode=btma_mode)

    summary_row = summarize_case(
        em,
        case_name,
        n,
        blockage_mode=blockage_mode,
        blocked_fraction=blocked_fraction,
        mac_result=mac_result,
    )
    node_df = make_node_dataframe(
        em,
        dto,
        case_name,
        n,
        blockage_mode=blockage_mode,
        mac_result=mac_result,
    )

    return summary_row, node_df


# =============================================================================
# Plotting
# =============================================================================

def save_pdf(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric(summary, col, ylabel, filename, include_no_mac=False, ylim=None):
    if include_no_mac:
        df = summary.copy()
    else:
        df = summary[summary["case"] != "No MAC overhead"].copy()

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for case_name, group in df.groupby("case"):
        group = group.sort_values("N")
        ax.plot(group["N"], group[col], marker="o", linewidth=2, label=case_name)

    
    ax.set_xlabel(r"$N_{\mathrm{SN}}$")
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()

    save_pdf(fig, filename)


def plot_total_network_energy(summary):
    plot_metric(
        summary,
        col="E_day_J_total",
        #title=r"Total daily network energy consumption",
        ylabel=r"$E_{\mathrm{CONS}}^{\mathrm{net}}$ [J]",
        filename="total_network_energy_per_day",
        include_no_mac=True,
    )


def plot_energy_per_delivered_packet(summary):
    plot_metric(
        summary,
        col="Energy_Per_Delivered_Packet_J",
        #title=r"Energy per delivered packet",
        ylabel=r"$E_{\mathrm{pkt}}$ [J]",
        filename="energy_per_delivered_packet",
        include_no_mac=True,
    )

def plot_average_cycle_energy(summary):
    plot_metric(
        summary,
        col="E_cycle_J_mean",
        ylabel=r"$\bar{E}_{\mathrm{cycle}}$ [J]",
        filename="average_cycle_energy",
        include_no_mac=True,
    )


def plot_mac_pdr(summary):
    plot_metric(
        summary,
        col="PDR_MAC",
        #title=r"MAC packet delivery ratio",
        ylabel=r"$\mathrm{PDR}$",
        filename="mac_pdr",
        include_no_mac=False,
        ylim=(-0.02, 1.05),
    )


def plot_mac_blocked_packets(summary):
    plot_metric(
        summary,
        col="MAC_Blocked_Packets_Per_Generated_Packet",
        #title=r"Blocked-packet fraction",
        ylabel=r"$p_{\mathrm{blk}}$",
        filename="mac_blocked_packets",
        include_no_mac=False,
        ylim=(-0.02, 1.05),
    )


def plot_cca_attempts(summary):
    plot_metric(
        summary,
        col="CCA_Attempts_Per_Generated_Packet",
        #title=r"Mean CCA attempts per generated packet",
        ylabel=r"$\bar{n}_{\mathrm{CCA}}$",
        filename="cca_attempts_per_generated_packet",
        include_no_mac=False,
    )


def plot_backoff_slots(summary):
    plot_metric(
        summary,
        col="Backoff_Slots_Per_Generated_Packet",
        #title=r"Mean backoff slots per generated packet",
        ylabel=r"$\bar{n}_{\mathrm{bo}}$",
        filename="backoff_slots_per_generated_packet",
        include_no_mac=False,
    )
    
def plot_backoff_and_cca_dual_axis(summary):
    df = summary[summary["case"] != "No MAC overhead"].copy()

    fig, ax1 = plt.subplots(figsize=(6.4, 4.4))
    ax2 = ax1.twinx()

    lines = []
    labels = []

    for case_name, group in df.groupby("case"):
        group = group.sort_values("N")

        l1, = ax1.plot(
            group["N"],
            group["Backoff_Slots_Per_Generated_Packet"],
            marker="o",
            linewidth=2,
            linestyle="-",
            label=case_name + r" -- $\bar{n}_{\mathrm{bo}}$",
        )

        l2, = ax2.plot(
            group["N"],
            group["CCA_Attempts_Per_Generated_Packet"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label=case_name + r" -- $\bar{n}_{\mathrm{CCA}}$",
        )

        lines.extend([l1, l2])
        labels.extend([l1.get_label(), l2.get_label()])

    ax1.set_xlabel(r"$N_{\mathrm{SN}}$")
    ax1.set_ylabel(r"$\bar{n}_{\mathrm{bo}}$")
    ax2.set_ylabel(r"$\bar{n}_{\mathrm{CCA}}$")

    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(lines, labels, loc="best")

    save_pdf(fig, "backoff_slots_and_cca_attempts")


def plot_retransmissions_per_generated_packet(summary):
    plot_metric(
        summary,
        col="Retransmissions_Per_Generated_Packet",
        ylabel=r"$\bar{n}_{\mathrm{ret}}$",
        filename="retransmissions_per_generated_packet",
        include_no_mac=False,
    )


def plot_union_active_payload_throughput_norm(summary):
    plot_metric(
        summary,
        col="Union_Active_Payload_Throughput_Normalized",
        #title=r"Normalized union-active-time payload throughput",
        ylabel=r"$\eta_{\mathrm{UAT}}$",
        filename="normalized_union_active_payload_throughput",
        include_no_mac=False,
        ylim=(-0.02, 1.05),
    )

def plot_average_node_energy(summary):
    plot_metric(
        summary,
        col="E_day_J_mean",
        ylabel=r"$\bar{E}_{\mathrm{CONS}}^{\mathrm{SN}}$ [J]",
        filename="average_node_energy_per_day",
        include_no_mac=True,
    )

def plot_true_collision_rate(summary):
    plot_metric(
        summary,
        col="True_MAC_Collision_Rate",
        ylabel=r"$p_{\mathrm{col}}$",
        filename="true_mac_collision_rate",
        include_no_mac=False,
        ylim=(-0.02, 1.05),
    )


def plot_phy_ul_failure_rate(summary):
    plot_metric(
        summary,
        col="PHY_UL_Failure_Rate",
        ylabel=r"$p_{\mathrm{PHY,UL}}$",
        filename="phy_ul_failure_rate",
        include_no_mac=False,
        ylim=(-0.02, 1.05),
    )


def plot_phy_dl_failure_rate(summary):
    plot_metric(
        summary,
        col="PHY_DL_Failure_Rate",
        ylabel=r"$p_{\mathrm{PHY,DL}}$",
        filename="phy_dl_failure_rate",
        include_no_mac=False,
        ylim=(-0.02, 1.05),
    )


def make_all_plots(summary):
    plot_total_network_energy(summary)
    plot_average_node_energy(summary)
    plot_energy_per_delivered_packet(summary)
    plot_mac_pdr(summary)
    plot_mac_blocked_packets(summary)
    plot_cca_attempts(summary)
    plot_backoff_slots(summary)
    plot_union_active_payload_throughput_norm(summary)
    plot_backoff_and_cca_dual_axis(summary)
    plot_retransmissions_per_generated_packet(summary)
    plot_average_cycle_energy(summary)
    plot_true_collision_rate(summary)
    plot_phy_ul_failure_rate(summary)
    plot_phy_dl_failure_rate(summary)
    


# =============================================================================
# Main
# =============================================================================

def main():
    plt.close("all")

    # The sweep requires the receiver-side collision fix in mac.py.
    import inspect
    channel_signature = inspect.signature(mac_module.VLC_Channel.__init__)
    if "ap_reachable" not in channel_signature.parameters:
        raise RuntimeError(
            "This sweep requires the updated mac.py with the ap_reachable "
            "argument in VLC_Channel."
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    summary_rows = []
    all_node_rows = []

    # BTMA is enabled internally for every CSMA/CA case. The legend reports
    # only the physical blockage condition, since BTMA is common to all cases.
    cases = [
        {
            "case_name": "No MAC overhead",
            "mac_enabled": False,
            "btma_mode": False,
            "blocked_fraction": 0.0,
            "blockage_mode": "none",
        },
        {
            "case_name": r"CSMA/CA",
            "mac_enabled": True,
            "btma_mode": True,
            "blocked_fraction": 0.0,
            "blockage_mode": "none",
        },
        {
            "case_name": r"CSMA/CA, $10\%$ UL+DL blocked",
            "mac_enabled": True,
            "btma_mode": True,
            "blocked_fraction": BLOCKED_FRACTION,
            "blockage_mode": "both",
        },
        {
            "case_name": r"CSMA/CA, $10\%$ DL-only blocked",
            "mac_enabled": True,
            "btma_mode": True,
            "blocked_fraction": BLOCKED_FRACTION,
            "blockage_mode": "downlink",
        },
    ]

    for n in NODE_SWEEP:
        print(f"\n=== N = {n} ===")

        for case in cases:
            print(f"  Running {case['case_name']}...")

            row, node_df = run_case(
                n=n,
                case_name=case["case_name"],
                mac_enabled=case["mac_enabled"],
                btma_mode=case["btma_mode"],
                blocked_fraction=case["blocked_fraction"],
                blockage_mode=case["blockage_mode"],
            )

            summary_rows.append(row)
            all_node_rows.append(node_df)

    summary = pd.DataFrame(summary_rows)
    per_node = pd.concat(all_node_rows, ignore_index=True)

    summary_path = RESULTS_DIR / "ir_pd_mac_energy_summary.csv"
    per_node_path = RESULTS_DIR / "ir_pd_mac_energy_per_node.csv"

    summary.to_csv(summary_path, index=False)
    per_node.to_csv(per_node_path, index=False)

    make_all_plots(summary)

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {per_node_path}")
    print("  plots/total_network_energy_per_day.pdf")
    print("  plots/average_node_energy_per_day.pdf")
    print("  plots/energy_per_delivered_packet.pdf")
    print("  plots/mac_pdr.pdf")
    print("  plots/mac_blocked_packets.pdf")
    print("  plots/cca_attempts_per_generated_packet.pdf")
    print("  plots/backoff_slots_per_generated_packet.pdf")
    print("  plots/normalized_union_active_payload_throughput.pdf")
    print("  plots/backoff_slots_and_cca_attempts.pdf")
    print("  plots/retransmissions_per_generated_packet.pdf")
    print("  plots/average_cycle_energy.pdf")
    print("  plots/true_mac_collision_rate.pdf")
    print("  plots/phy_ul_failure_rate.pdf")
    print("  plots/phy_dl_failure_rate.pdf")

    print("\nPreview:")
    display_cols = [
        "case",
        "N",
        "blockage_mode",
        "blocked_fraction",
        "PDR_AP",
        "PDR_MAC",
        "True_MAC_Collision_Rate",
        "PHY_UL_Failure_Rate",
        "PHY_DL_Failure_Rate",
        "E_day_J_total",
        "Energy_Per_Generated_Packet_J",
        "Energy_Per_Delivered_Packet_J",
        "CCA_Attempts_Per_Generated_Packet",
        "Backoff_Slots_Per_Generated_Packet",
        "MAC_Blocked_Packets_Per_Generated_Packet",
        "Union_Active_Payload_Throughput_Normalized",
        "Union_Active_Time_s",
        "Retransmissions_Per_Generated_Packet",
    ]

    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
