import numpy as np
import json
from dataclasses import asdict
import matplotlib.pyplot as plt
from enlight_iot import EnLightConfig, PhyNet, EnergyManager
from enlight_iot.core.interface import PhyResultsDTO  
from design_B1 import master_design_example 

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 10,
})

# Silence division-by-zero warnings for blocked optical paths
np.seterr(all='ignore') 

# ── 1. Helper to save Numpy Arrays to JSON ──
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle numpy arrays in the design dictionary."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ── 2. The Experiment Manager Wrapper ──
def run_experiment(design_dict, run_name="exp_001", run_budget=False, btma_mode=False, **kwargs):
    """Wraps the simulation modules and guarantees robust data saving."""

    print(f"Budget Run Enabled: {run_budget}")
    
    # ── A. Run the Physical Layer ──
    print(f"\n[{run_name}] Computing Physical Layer Matrices (Channel, Physics)...")
    
    global pn,em
    
    pn = PhyNet(design_dict, budget_run=run_budget, btma_mode = btma_mode, **kwargs)

    
    # Save the reporting matrices (Old method untouched for plotting)
    pn.save_phy_state(f"{run_name}_phy_matrices.npz")
    
    # ── B. Export and Save the Telemetry Explicitly ──
    dto = pn.export_energy_telemetry()
    telemetry_filepath = f"{run_name}_phy_telemetry.npz"
    dto.save_npz(telemetry_filepath)
    print(f"[{run_name}] Telemetry saved to {telemetry_filepath}")

    # ── C. Load Telemetry and Run Energy/MAC Simulator ──
    print(f"[{run_name}] Running Dual MAC Simulations & Energy Manager...")
    
    # Strictly load the DTO from disk to prove complete decoupling
    loaded_dto = PhyResultsDTO.load_npz(telemetry_filepath)
    
    # NOTE: verbose=False turns off the node-by-node warning prints in the terminal
    em = EnergyManager(loaded_dto, design_dict, MAC=False, btma_mode=btma_mode, **kwargs)
    
    # Extract Tabular Data
    results_df = em.get_results_df()
    
    # Extract Actual Transmit Powers used (Catches dynamic changes if run_budget=True)
    actual_ir_tx = pn.snm.OTx_elements.p.flatten().tolist() if pn.snm.ir_flag > 0 else []
    actual_rf_tx = pn.snm.RFTx_elements.p.flatten().tolist() if pn.snm.rf_flag > 0 else []

    # Compile the Master Metadata tracking dictionary
    experiment_metadata = {
        "experiment_name": run_name,
        "budget_run_enabled": run_budget,
        "actual_tx_powers_used": {
            "IR_nodes_watts": actual_ir_tx,
            "RF_nodes_dBm": actual_rf_tx
        },
        "original_user_design": design_dict,
        "full_system_config": asdict(pn.config) # <--- Captures every single default parameter!
    }

    # ── D. Save Final Results ──
    results_df.to_csv(f"{run_name}_results.csv", index=False)
    
    with open(f"{run_name}_metadata.json", 'w') as f:
        json.dump(experiment_metadata, f, indent=4, cls=NumpyEncoder)
        
    print(f"[{run_name}] Data saved successfully -> .csv, .npz (x2), and .json")
    return results_df

# ── 3. Your Main Execution Logic ──
def run_system_test():
    print("=" * 60)
    print(" Booting EnLight IoT Framework - System Verification")
    print("=" * 60)

    # ── Execute the Simulation via the Wrapper ──
    print("\n[1/3] Loading Configuration...")
    config = EnLightConfig()

    # The wrapper handles initialization and ALL the safe saving mechanics
    df = run_experiment(
        design_dict=master_design_example,
        run_name="experiment_1",
        run_budget=False,
        btma_mode=True,
        config=config # Explicitly pass config to prevent any positional bugs down the line
    )

    print("\n" + "=" * 60)
    print(" SYSTEM TEST SUCCESSFUL: All modules imported and executed decoupled.")
    print("=" * 60)

if __name__ == "__main__":
    run_system_test()
    
    
    #plot SNR across the floor 
    # --- 1. Define physical room dimensions (in meters) ---
    room_length_x = 5.0
    room_width_y = 5.0

    # --- 2. Load the telemetry data ---
    #data = np.load('experiment_1_phy_telemetry.npz')
    snr_downlink = pn.snr_d_dB
    

    # Font sizes
    LABEL_SIZE = 16
    TICK_SIZE  = 16

    room_length_x = 5.0
    room_width_y  = 5.0

    
   

    num_sensors = len(snr_downlink)
    grid_side   = int(np.sqrt(num_sensors))

    if grid_side**2 != num_sensors:
        print(f"Warning: {num_sensors} sensors do not form a perfect square.")
    else:
        snr_grid        = snr_downlink.reshape((grid_side, grid_side))
        masked_snr_grid = np.ma.masked_where(snr_grid <= -1, snr_grid)

        cmap = plt.cm.turbo.copy()
        cmap.set_bad(color='black')

        plt.figure(figsize=(8, 6))

        im = plt.imshow(
            masked_snr_grid,
            cmap=cmap,
            origin='lower',
            extent=[0, room_length_x, 0, room_width_y],
            vmax = 55.4,
            vmin = 30
        )

        cbar = plt.colorbar(im)
        cbar.set_label('$SNR_d$ [dB]', fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

        plt.xlabel('$x$ [m]', fontsize=LABEL_SIZE)
        plt.ylabel('$y$ [m]', fontsize=LABEL_SIZE)
        plt.tick_params(axis='both', labelsize=TICK_SIZE)

        plt.tight_layout()
        plt.savefig("plots/b1/snr_d_pd_b.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        
    #plot required power

    # Font sizes
    LABEL_SIZE = 16
    TICK_SIZE  = 16

    room_length_x = 5.0
    room_width_y  = 5.0

   
   

    num_sensors = len(snr_downlink)
    grid_side   = int(np.sqrt(num_sensors))
    
    snr_d_diff = pn.snr_d_diff_dB

    if grid_side**2 != num_sensors:
        print(f"Warning: {num_sensors} sensors do not form a perfect square.")
    else:
        snr_grid        = snr_d_diff.reshape((grid_side, grid_side))
        masked_snr_grid = np.ma.masked_where(snr_grid <= -1, snr_grid)

        cmap = plt.cm.plasma.copy()
        cmap.set_bad(color='black')

        plt.figure(figsize=(8, 6))

        im = plt.imshow(
            masked_snr_grid,
            cmap=cmap,
            origin='lower',
            extent=[0, room_length_x, 0, room_width_y],
            vmin = 30
        )

        cbar = plt.colorbar(im)
        cbar.set_label('$SNR_d$ [dB]', fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

        plt.xlabel('$x$ [m]', fontsize=LABEL_SIZE)
        plt.ylabel('$y$ [m]', fontsize=LABEL_SIZE)
        plt.tick_params(axis='both', labelsize=TICK_SIZE)

        plt.tight_layout()
        plt.savefig("plots/b1/snr_d_diff_b.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        
        
    room_length_x = 5.0
    room_width_y = 5.0

    # --- 2. Load the telemetry data ---
    #data = np.load('experiment_1_phy_telemetry.npz')
    snr_uplink = pn.snr_u_dB
    

    # Font sizes
    LABEL_SIZE = 16
    TICK_SIZE  = 16

    room_length_x = 5.0
    room_width_y  = 5.0

    
   

    num_sensors = len(snr_downlink)
    grid_side   = int(np.sqrt(num_sensors))

    if grid_side**2 != num_sensors:
        print(f"Warning: {num_sensors} sensors do not form a perfect square.")
    else:
        snr_grid        = snr_uplink.reshape((grid_side, grid_side))
        masked_snr_grid = np.ma.masked_where(snr_grid <= -0, snr_grid)

        cmap = plt.cm.cividis.copy()
        cmap.set_bad(color='black')

        plt.figure(figsize=(8, 6))

        im = plt.imshow(
            masked_snr_grid,
            cmap=cmap,
            origin='lower',
            extent=[0, room_length_x, 0, room_width_y]
        )

        cbar = plt.colorbar(im)
        cbar.set_label('$SNR_u$ [dB]', fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

        plt.xlabel('$x$ [m]', fontsize=LABEL_SIZE)
        plt.ylabel('$y$ [m]', fontsize=LABEL_SIZE)
        plt.tick_params(axis='both', labelsize=TICK_SIZE)

        plt.tight_layout()
        plt.savefig("plots/b1/snr_u_b.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        
    #plot required power

    # Font sizes
    LABEL_SIZE = 16
    TICK_SIZE  = 16

    room_length_x = 5.0
    room_width_y  = 5.0

   
   

    num_sensors = len(snr_downlink)
    grid_side   = int(np.sqrt(num_sensors))
    
    snr_u_diff = pn.snr_u_diff_dB

    if grid_side**2 != num_sensors:
        print(f"Warning: {num_sensors} sensors do not form a perfect square.")
    else:
        snr_grid        = snr_u_diff.reshape((grid_side, grid_side))
        masked_snr_grid = np.ma.masked_where(snr_grid <= -1, snr_grid)

        cmap = plt.cm.turbo.copy()
        cmap.set_bad(color='black')

        plt.figure(figsize=(8, 6))

        im = plt.imshow(
            masked_snr_grid,
            cmap=cmap,
            origin='lower',
            extent=[0, room_length_x, 0, room_width_y],
            vmin = 4,
            vmax = 9
        )

        cbar = plt.colorbar(im)
        cbar.set_label('$SNR_u$ [dB]', fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

        plt.xlabel('$x$ [m]', fontsize=LABEL_SIZE)
        plt.ylabel('$y$ [m]', fontsize=LABEL_SIZE)
        plt.tick_params(axis='both', labelsize=TICK_SIZE)

        plt.tight_layout()
        plt.savefig("plots/b1/snr_u_diff_b.pdf", format="pdf", bbox_inches="tight")
        plt.show()