from pathlib import Path
from pyenlight.core.config import EnLightConfig
from pyenlight.network.orchestrator import PhyNet
from pyenlight.hardware.energy import EnergyManager

from design_example import design


# --------------------------------------------------
# Output directory
# --------------------------------------------------

OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

config = EnLightConfig()


# --------------------------------------------------
# 1. Physical-layer simulation
# --------------------------------------------------

phy = PhyNet(
    design,
    budget_run=False,
    config=config,
    btma_mode=True,
)

phy.save_phy_state(
    OUT / "experiment_phy_matrices.npz"
)


# --------------------------------------------------
# 2. Export the PHY-to-MAC/energy interface
# --------------------------------------------------

phy_data = phy.export_energy_telemetry()

phy_data.save_npz(
    OUT / "experiment_phy_telemetry.npz"
)


# --------------------------------------------------
# 3. Energy calculation without MAC
# --------------------------------------------------

energy_no_mac = EnergyManager(
    phy_data=phy_data,
    design=design,
    config=config,
    MAC=False,
    btma_mode=True,
    MAC_mode="unslotted",
)

energy_no_mac.save_csv(
    OUT / "experiment_no_mac.csv"
)


# --------------------------------------------------
# 4. MAC simulation and MAC-aware energy calculation
# --------------------------------------------------

energy_mac = EnergyManager(
    phy_data=phy_data,
    design=design,
    config=config,
    MAC=True,
    btma_mode=True,
    MAC_mode="unslotted",
)

energy_mac.save_csv(
    OUT / "experiment_mac.csv"
)


# --------------------------------------------------
# 5. Access the results
# --------------------------------------------------

results_no_mac = energy_no_mac.get_results_df()
results_mac = energy_mac.get_results_df()

print(results_no_mac.head())
print(results_mac.head())