# pyenlight

## Description

`pyenlight` is a modular cross-layer simulation framework for indoor optical and hybrid optical/RF Internet-of-Things (IoT) networks. It combines:

- three-dimensional room and node geometry;
- visible-light communication (VLC) downlinks;
- infrared (IR) or radio-frequency (RF) uplinks;
- line-of-sight (LoS), diffuse, and reconfigurable intelligent surface (RIS) optical paths;
- ambient artificial and natural light;
- human-body blockage;
- photodiode (PD) and photovoltaic (PV) optical receivers;
- physical-layer SNR and BER evaluation;
- IEEE 802.15.4/802.15.7-based CSMA/CA simulation;
- optional busy-tone multiple access (BTMA);
- hidden-node and physical-link failures;
- device-level energy consumption;
- PV energy harvesting and battery-lifetime estimation.

The library is intended for cross-layer studies in which physical-layer propagation, receiver hardware, medium access, and sensor-node energy consumption must be evaluated consistently.

The principal workflow is:

1. define a design dictionary;
2. create a `PhyNet` object to evaluate the PHY;
3. export the required PHY telemetry;
4. create an `EnergyManager`, optionally with MAC enabled;
5. obtain per-node and aggregate results.

---

The Python package is imported as `pyenlight`.

## General Information

pyenlight is a simulation library. Scenarios are defined through Python design dictionaries containing the room geometry, optical surfaces, nodes, hardware parameters, communication settings, MAC configuration, and energy profile.

The repository also contains generated results for the example experiments. These include detailed PHY matrices, compact PHY information to be used in the energy/MAC layers, experiment metadata, per-node CSV results, and figures. These outputs should be interpreted together with the design dictionary, configuration, software revision, MAC seeds, simulation duration, and PHY/MAC thresholds used to generate them.

### Acknowledgment

This work was funded by the European Union under the Marie Skłodowska-Curie Doctoral Network **OWIN6G: Optical Wireless Sensor Networks for 6G (Grant agreement ID: 101119624)**.

## Usage Instructions

### Installation

The package requires Python 3.8 or later.

```bash
pip install -e .
```

The current `setup.py` declares:

```text
numpy
scipy
matplotlib
simpy
pandas
```

Linux installation in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```
Using a virtual / conda environment is strongly recommended.

Windows installation in a virtual environment with PowerShell:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### Main Simulation Workflow

A minimal high-level workflow is:

```python
from pyenlight.core.config import EnLightConfig
from pyenlight.network.orchestrator import PhyNet
from pyenlight.hardware.energy import EnergyManager
import numpy as np

config = EnLightConfig()

design = {
    "environment": {
        "dimensions": [5.0, 5.0, 3.0], # L, W, H in meters
        "wall_resolution": [20, 20],   # Grid patches for diffuse reflections
        "reflectivity": {
            "floor": 0.2,
            "ceiling": 0.6,
            "walls": 0.8
        },
    },
    
    "nodes": {
        "masters": {
            "positions": np.array([[2.5, 2.5, 3.0]]), # Ceiling center
            "nT": [0, 0, -1],   # TX pointing down
            "nR": [0, 0, -1],   # RX pointing down
            "rx_area": 1e-4,    # 1 cm^2
            "m": 1,             # Lambertian order
            "FOV": np.pi/2,      # 90 degrees
            "tx_power": 1.0,    # Downlink VLC Power (W)
            "sensitivity": -100,
            "IR_pass_filter": True
        },
        "sensors": {
            "positions": np.array([[2,2,0], [3,3,0]]), 
        },

    },
}

phy = PhyNet(
    design,
    budget_run=False,
    config=config,
    btma_mode=True,
)

telemetry = phy.export_energy_telemetry()

energy = EnergyManager(
    phy_data=telemetry,
    design=design,
    config=config,
    MAC=True,
    btma_mode=True,
    MAC_mode="unslotted",
)

results = energy.get_results_df()

```

This is a very simple design, where the missing parameters are fetched from EnLightConfig.
It is recommended to start with a validated scenario file such as `design_A.py` to check if everything executes as expected. 
`design_example.py` is an example of a fully configured design dictionary. Use it as a blueprint to define your scenario if you need full control.

More details about running experiments can be found on the user's guide.
More details about the code implementation can be found on the developer's guide.


---
#### Execution sequence inside `PhyNet`

The `PhyNet` constructor performs the following operations:

1. loads the simulation parameters;
2. constructs the room using `RoomBuilder` and `Room`;
3. parses sensor and master-node definitions;
4. parses ambient optical sources;
5. creates sensor, master, and ambient-node managers;
6. computes LoS, diffuse, RIS, RF, and ambient gains;
7. computes receiver noise and bandwidths;
8. optionally determines required Tx power when `budget_run=True`;
9. computes received powers and currents;
10. computes SNR and related PHY metrics.

Thus, creating `PhyNet` is not a lightweight configuration operation. It executes the complete PHY calculation.

---



---
### Output Organization

A reproducible experiment should store at least:

```text
experiment_name_metadata.json
experiment_name_phy_matrices.npz
experiment_name_phy_telemetry.npz
experiment_name_results.csv
```

File details:

| File | Purpose |
|---|---|
| metadata JSON | design, configuration, seed, and run metadata |
| PHY matrices NPZ | detailed channel and PHY arrays |
| PHY telemetry NPZ | compact PHY-to-energy/MAC interface |
| results CSV | per-node and aggregate output |

The PHY matrices and energy results are separate artifacts. They may be stored in the same experiment directory, but they are not produced as one combined file.

---
## Requirements

The current version was tested with Python 3.11 and the following package versions:

```text
numpy==2.4.6
scipy==1.17.1
matplotlib==3.11.0
simpy==4.1.2
pandas==3.0.3
```

These exact versions are recorded in `requirements-lock.txt` for reproducibility. The package may also work with other compatible versions.

## Code Information

### Package Structure

```text
pyenlight/
├── core/
│   ├── config.py
│   ├── interface.py
│   └── utils.py
├── environment/
│   ├── geometry.py
│   └── channel.py
├── hardware/
│   ├── devices.py
│   ├── energy.py
│   └── physics.py
├── network/
│   ├── nodes.py
│   ├── orchestrator.py
│   └── mac.py
└── __init__.py
```

### Repository Files

The repository also includes the configuration files, execution scripts, and outputs used for the paper experiments. The files `design_A.py`, `design_B.py`, `design_C.py`, and `design_D.py` define the corresponding simulation scenarios, while `run_A.py`, `run_B.py`, `run_C.py`, and `run_D.py` execute them. The `run_MAC.py` script performs the MAC-specific simulations. The directories `plots/A`, `plots/B`, `plots/C`, and `plots/D` contain the corresponding figures, while the `MAC` directory contains the MAC-specific results and plots. 
The generated data are stored inside the `data` folder. The subdirectories `results_experiment_A`, `results_experiment_B`, `results_experiment_C`, and `results_experiment_D` contain the saved configurations, PHY results, PHY DTO, and energy/MAC outputs generated for each scenario.

The `bw_d.npy` file contains the bandwidth of the PV-based receivers in Scenario D. The downlink data rate is not automatically limited by the PV receiver bandwidth. Therefore, after each simulation involving PV-based receivers, the required bandwidth for the selected data rate should be compared with the available PV bandwidth. It is easy to generate these data by running the corresponding scenarios.

#### `core`

The `core` package contains global configuration objects, shared utilities, and data-transfer structures.

- `config.py`: dataclass-based default parameters.
- `interface.py`: `PhyResultsDTO`, which transfers PHY outputs to the energy and MAC layers.
- `utils.py`: array conversion, geometry, probability, and helper functions.

#### `environment`

The `environment` package constructs the room and evaluates propagation.

- `geometry.py`: room surfaces, RISs, windows, blockers, and optical/RF element containers.
- `channel.py`: LoS, diffuse, RIS, and RF channel-gain calculations.

#### `hardware`

The `hardware` package describes optical/electrical devices and sensor-node energy.

- `physics.py`: optical spectra, filters, detector responses, and effective responsivity.
- `devices.py`: TIA, IR driver, RF transmitter-current model, and PV receiver.
- `energy.py`: cycle energy, MAC-aware state durations, harvesting, and battery lifetime.

#### `network`

The `network` package creates nodes, orchestrates PHY evaluation, and simulates medium access.

- `nodes.py`: parsing and validation of master, sensor, and ambient-node definitions.
- `orchestrator.py`: `PhyNet`, the main physical-layer simulation entry point.
- `mac.py`: slotted and unslotted CSMA/CA, ACKs, hidden nodes, BTMA, and aggregation.

---
### Constants and Default Parameters

Default constants and fallback values are stored in `EnLightConfig`, which contains six dataclasses:

```python
config.physics
config.spectral
config.env
config.hardware
config.comm
config.devices
```

Values defined explicitly in the design dictionary generally override the corresponding defaults.

#### Physical constants

`PhysicsConfig` includes:

| Parameter | Meaning | Default |
|---|---|---:|
| `q` | elementary charge | `1.60217663e-19` C |
| `kB` | Boltzmann constant | `1.380649e-23` J/K |
| `c0` | speed of light | `299792458.0` m/s |
| `hP` | Planck constant | `6.62607015e-34` J s |
| `T0` | reference temperature | `300.0` K |
| `pd_peak` | peak solar spectral irradiance | `2e9` W/m$^2$/m |
| `T` | device temperature | `298` K |
| `eo` | vacuum permittivity | `8.854e-12` F/m |

It also defines the six Cartesian unit vectors `xp`, `xm`, `yp`, `ym`, `zp`, and `zm`.

#### Spectral parameters

`SpectralConfig` defines the integration range and resolution:

| Parameter | Meaning | Default |
|---|---|---:|
| `l_min` | minimum wavelength | `200e-9` m |
| `l_max` | maximum wavelength | `1300e-9` m |
| `grid_points` | wavelength samples | `1000` |
| `t_sun` | solar black-body temperature | `5800` K |

#### Environment parameters

`EnvironmentConfig` provides geometry and optical defaults:

| Parameter | Meaning | Default |
|---|---|---:|
| `reflectivity` | default surface reflectivity | `0.6` |
| `wall_resolution` | wall discretization | `(20, 20)` |
| `m` | Lambertian order | `1` |
| `rx_area` | optical Rx area | `1e-4` m² |
| `fov` | receiver field of view | `pi/2` |
| `bounces` | diffuse-reflection order | `4` |
| `room_dim` | default room dimensions | `[5, 5, 3]` m |
| `ris_element_area` | RIS-element area | `1e-4` m² |

Increasing wall resolution or the number of diffuse bounces can substantially increase memory use and runtime.

#### Hardware parameters

`HardwareConfig` contains MCU, receiver, and battery defaults:

| Parameter | Meaning | Default |
|---|---|---:|
| `f_mcu` | MCU clock frequency | `16e6` Hz |
| `f_s` | sensing rate | `1e3` Hz |
| `voltage` | node operating voltage | `3.3` V |
| `I_mcu` | MCU active current | `2.73e-3` A |
| `I_adc` | ADC current | `0.7e-3` A |
| `I_ext` | external sensor current | `1.0e-3` A |
| `I_sleep` | sleep current | `2e-6` A |
| `I_wake` | wake current | `1e-3` A |
| `I_tia` | TIA current | `0.7e-3` A |
| `battery_capacity_mAh` | battery capacity | `500` mAh |
| `initial_soc` | initial state of charge | `1.0` |
| `V_batt` | battery voltage | `3.6` V |
| `mpp_eff` | MPP conversion efficiency | `0.8` |

#### Communication parameters

`CommConfig` contains default Tx powers, data sizes, rates, and cycle timings:

| Parameter | Meaning | Default |
|---|---|---:|
| `IR_Tx_power` | IR optical Tx power | `15e-3` W |
| `VLC_Tx_power` | VLC optical Tx power | `1.0` W |
| `sensitivity` | RF receiver sensitivity | `-100` dBm |
| `uplink_type` | `0`: IR, `1`: RF | `0` |
| `rx_type` | `0`: PD, `1`: PV | `0` |
| `L_up_bits` | uplink payload length | `1024` bits |
| `L_dw_bits` | downlink/ACK length | `128` bits |
| `bit_rate_up_ir` | IR uplink data rate | `10e3` bit/s |
| `bit_rate_up_rf` | RF uplink data rate | `250e3` bit/s |
| `bit_rate_dw` | VLC downlink data rate | `10e3` bit/s |
| `t_init` | initialization duration | `5e-3` s |
| `t_wait` | turnaround/wait duration | `1e-3` s |
| `T_cycle` | node operation period | `1.0` s |
| `harvesting_hours` | daily harvesting interval | `5.0` h |
| `n_sp` | ON/OFF keying spectral efficiency | `0.4` |

#### Device dictionaries

`DeviceConfig` stores parameter dictionaries for:

- RF propagation;
- the TIA;
- the PV equivalent circuit;
- the IR driver;
- the RF transmitter-current model.

---
### Design Dictionary

The design dictionary describes one simulation scenario. Its principal top-level sections are:

```python
design = {
    "environment": {...},
    "nodes": {...},
    "PV_circuit": {...},
    "TIA": {...},
    "energy_profile": {...},
    "protocol": {...},
    "MPP": {...},
}
```

Not every section is required. Missing values are generally taken from `EnLightConfig`.

#### Environment definition

A representative structure is:

```python
design["environment"] = {
    "dimensions": [5.0, 5.0, 3.0],
    "wall_resolution": [20, 20],
    "reflectivity": {
        "floor": 0.2,
        "ceiling": 0.6,
        "walls": 0.8,
    },
    "special_surfaces": [
        {
            "type": "window",
            "name": "west_window",
            "center": [0.0, 2.5, 1.5],
            "dims": [1.0, 1.0],
            "const_axis": 0,
            "resolution": [10, 10],
            "normal": [1.0, 0.0, 0.0],
            "reflectivity": 0.05,
        },
        {
            "type": "RIS",
            "name": "ris_1",
            "center": [5.0, 2.5, 1.5],
            "dims": [1.0, 1.0],
            "const_axis": 0,
            "resolution": [10, 10],
            "normal": [-1.0, 0.0, 0.0],
            "reflectivity": 0.9,
        },
    ],
    "blockers": {
        "positions": [
            [4,1,0],
            [3.5,3.5,0]
            ],
         "radius": 0.3,       
         "height":1.7    
        } ,
}
```

#### Surface axes

`const_axis` selects the coordinate that remains constant:

- `0`: plane normal to the x-axis;
- `1`: plane normal to the y-axis;
- `2`: plane normal to the z-axis.

Surface points are placed at patch centers. The patch area is computed from the two surface dimensions and their respective resolutions.

#### Node definitions

The `nodes` section may contain:

```python
design["nodes"] = {
    "masters": {...},
    "sensors": {...},
    "ambient_nodes": {...},
}
```

Each node group is parsed by `NodeBuilder`. Parameters may be scalars or arrays. Scalars are expanded to the number of nodes by the utility functions.

The node definitions include, depending on node type:

- position vectors;
- optical Tx and Rx normal vectors;
- optical Tx power;
- Lambertian order;
- receiver area;
- field of view;
- receiver type;
- uplink type;
- RF Tx power;
- data rates;
- optical source and detector names;
- filter names;
- sensitivity and spreading parameters.

Use arrays of shape `(N, 3)` for positions and orientation vectors. Per-node scalar parameters should have length `N` or be provided as a single scalar if they are the same for all nodes.

#### Receiver and uplink-type flags

The current implementation uses numerical flags:

```text
rx_type = 0 -> photodiode receiver
rx_type = 1 -> photovoltaic receiver

uplink_type = 0 -> infrared optical uplink
uplink_type = 1 -> RF uplink
```

Mixed sensor populations are supported through per-node arrays.

---
### Geometry Model

#### Element containers

The geometry layer uses vectorized element classes:

- `Elements`;
- `OpticalRxElements`;
- `OpticalTxElements`;
- `RFTxElements`.

Each object represents a batch of elements rather than a single device. The central fields are:

```text
r : element positions, shape (N, 3)
n : orientation vectors, shape (N, 3)
N : number of elements
```

Optical receivers additionally contain area, field of view, reflectivity, and receiver type. Optical transmitters contain power and Lambertian order.

The `Elements.merge()` method combines multiple compatible batches using vertical stacking.

#### Room construction

`RoomBuilder` reads dimensions, resolution, reflectivity, special surfaces, and blockers. `Room` then generates:

- floor;
- ceiling;
- west wall;
- east wall;
- south wall;
- north wall;
- optional windows;
- optional RIS surfaces.

The room boundary surfaces are discretized into patches. These patches act as intermediate transmitters and receivers for diffuse propagation.

#### Special-surface overlap

When a window or RIS is added to a wall, overlapping wall patches are handled by the room-building logic. This prevents the original wall and the inserted special surface from occupying the same effective area.

#### Human blockers

Blockers are passed from the design dictionary to `ChannelEngine`. The implemented model treats blockers as finite vertical bodies and determines whether a Tx–Rx segment intersects them.

Blockage is applied in the LoS channel calculation. Consequently, a blocked direct link may still retain diffuse or RIS-assisted power unless these paths are also geometrically excluded.

---
### Optical and RF Channel Models

`ChannelEngine` provides four principal methods:

```python
los_gains(tx, rx)
diffuse_gains(tx, rx, bounces)
ris_gains(tx, rx)
rf_gains(tx, rx)
```

#### LoS optical gain

The LoS calculation uses a Lambertian emission model. It considers:

- Tx–Rx distance;
- Tx irradiance angle;
- Rx incidence angle;
- Tx Lambertian order;
- receiver area;
- receiver field of view;
- blockage.

Inputs and outputs are vectorized. The resulting matrix contains one gain for each Tx–Rx pair.

#### Diffuse optical gain

Diffuse gain is evaluated through the discretized room surfaces. The configured number of reflections is controlled by:

```python
config.env.bounces
```

Higher reflection orders increase complexity. For dense wall grids, the intermediate channel matrices can become the dominant memory cost.

#### RIS gain

RIS gain is computed as a two-hop optical path through the RIS elements:

```text
Tx -> RIS element -> Rx
```

The RIS geometry, element normals, reflectivity, and element area determine the contribution. The result is added to the LoS and diffuse components.

#### RF gain

The RF uplink uses a configurable indoor path-loss model. Its default parameter dictionary includes:

```python
{
    "n": 1.46,
    "pl_ref": 34.62,
    "k": 2.03,
    "f": 2.45,
    "sigma": 3.76,
    "sigma_factor": 2,
}
```

RF gain is represented in the units expected by the RF power-budget calculations, including dB-domain operations in `PhyNet`.

---
### Spectral Model

`SpectralPhysics` calculates effective source–detector responsivity from spectral overlap.

It provides models for:

- a white LED;
- a TSFF5210-based IR emitter;
- sunlight;
- photodiode responsivity;
- solar-panel sensitivity;
- optical filters.

The effective responsivity is obtained by integrating the product of:

```text
source spectrum × detector response × filter transmission
```

This permits the same geometric optical power to produce different electrical currents for different sources, filters, and detectors.

Primary methods include:

```python
white_led_spectrum(wl)
tsff5210_spectrum(wl)
sun_spectrum(wl)
photodiode_responsivity(wl)
solar_panel_sensitivity(wl)
calculate_effective_responsivity(...)
get_responsivity_by_name(...)
```

The wavelength interval and number of samples are controlled by `SpectralConfig`.

---
### Node Managers

#### `NodeBuilder`

`NodeBuilder` extracts a node group from the design dictionary and normalizes its values. It performs:

- fallback to global defaults;
- scalar-to-array expansion;
- position and orientation reshaping;
- node-count inference;
- basic dimensional checks;
- filtering of optical fields for hybrid IR/RF populations.

Call:

```python
builder = NodeBuilder(design, "sensors", config)
```

Valid node-group names used by the orchestrator are:

```text
sensors
masters
ambient_nodes
```

#### `SNManager`

`SNManager` creates sensor-node optical and RF element groups and determines:

- number of sensors;
- IR and RF masks;
- uplink data rates;
- receiver types;
- spectral effective responsivity;
- TIA configuration;
- optical Tx, optical Rx, and RF Tx elements.

It supports mixed IR/RF uplinks and mixed PD/PV receivers.

#### `MNManager`

`MNManager` creates master-node optical Tx and Rx structures. It provides the VLC downlink source, IR uplink receiver, and RF receiver parameters.

#### `ANManager`

`ANManager` represents artificial ambient optical sources. Ambient nodes contribute noise and, depending on receiver type, may also contribute harvestable optical power.

---
### Physical-Layer Orchestration

#### `oPhyGains`

`oPhyGains` is the gain and received-current engine used by `PhyNet`. It calculates:

##### Downlink

```text
master optical Tx -> sensor optical Rx
```

Components:

- `h_d_los`;
- `h_d_diff`;
- `h_d_ris`;
- received optical powers;
- received electrical currents.

##### IR uplink

```text
sensor IR Tx -> master optical Rx
```

Components:

- `h_u_los`;
- `h_u_diff`;
- `h_u_ris`;
- received optical powers;
- received electrical currents.

##### Sensor-to-sensor optical links

When BTMA is disabled, sensor-to-sensor gains are calculated for CCA and hidden-node detection.

##### RF uplink

For RF sensors, the engine calculates:

- sensor-to-master RF gain;
- sensor-to-sensor RF gain.

##### Ambient illumination

Artificial sources and windows are evaluated for both sensor and master receivers. The resulting currents are used in shot-noise and PV calculations.

#### `PhyNet`

`PhyNet` is the main PHY API:

```python
phy = PhyNet(
    design,
    budget_run=False,
    config=config,
    btma_mode=True,
)
```

Parameters:

| Argument | Meaning |
|---|---|
| `design` | complete scenario dictionary |
| `budget_run` | determine required Tx power and orientation (towards CN) when true |
| `config` | optional `EnLightConfig` |
| `btma_mode` | enable BTMA assumptions and omit optical sensor-to-sensor CCA gains |

##### Power-budget mode

When `budget_run=True`, `PhyNet` may modify Tx powers:

- IR nodes are aligned toward the master when a single master exists;
- minimum IR optical Tx power is estimated from a target BER;
- minimum RF Tx power is estimated from receiver sensitivity.

The default optical target BER used by `set_tx_power()` is `3.8e-3`.

##### Multiple masters

The current alignment method only supports a single master. When multiple masters are present, automatic IR orientation is not implemented.

---
### Receiver Noise and Communication Metrics

#### Downlink bandwidth

The downlink receiver bandwidth is calculated from:

```text
BW_d = Rb_d / n_sp_d
```

#### Uplink bandwidth

For IR uplinks:

```text
BW_u = Rb_up_ir / n_sp_u
```
where $R_b$ is the data rate and $n_{sp}$ is the modulation spectral-efficiency factor, set to 0.4 for OOK.

#### PD receiver noise

PD receivers include:

- TIA noise;
- shot noise from sunlight;
- shot noise from artificial ambient sources.

The TIA is evaluated by `TIA.calc_noise_power()`.

#### PV receiver model

PV receivers use the `PV` class and a five-parameter equivalent-circuit representation. The model includes:

- DC operating point;
- photocurrent;
- series and shunt resistance;
- junction and diffusion capacitance;
- frequency-dependent transfer function;
- thermal noise;
- shot noise;
- signal voltage.

The PV receiver bandwidth limits the usable downlink rate. PV parameters are taken from `config.devices.pv_circuit` and may be overridden by `design["PV_circuit"]`.

#### SNR components

`PhyNet.compute_metrics()` derives total and component metrics, including:

- total downlink SNR;
- LoS-only downlink SNR;
- diffuse-only downlink SNR;
- RIS-only downlink SNR;
- IR uplink SNR;
- sensor-to-sensor SNR;
- RF link metrics.

The implementation uses chunked processing for selected calculations to reduce peak memory use.

#### BER relation

The library utilities use the Gaussian Q-function and inverse Q-function. For OOK-style evaluation, the required SNR is obtained through the target Q-function argument, and the optical Tx-power budget is computed accordingly.

---
### PHY Telemetry

`PhyResultsDTO` is the interface between `PhyNet` and `EnergyManager`.

Its fields are:

```python
no_sensors
rb_up
rb_down
flag_pv
uplink_type
otx_p
rftx_p
snr_d_dB
snr_ss_dB
snr_u_dB
phy_pdr_up_rf
hidden_node_mask_rf
pv_v_active
pv_i_active
```

Create telemetry using:

```python
telemetry = phy.export_energy_telemetry()
```

Save it using:

```python
telemetry.save_npz("experiment_1_phy_telemetry.npz")
```

Reload it using:

```python
from pyenlight.core.interface import PhyResultsDTO

telemetry = PhyResultsDTO.load_npz(
    "experiment_1_phy_telemetry.npz"
)
```

The loader converts `no_sensors` from a zero-dimensional NumPy array to a Python integer.

`PhyNet.save_phy_state()` separately stores larger physical-layer matrices and isolated link components. This is useful when detailed channel results are needed without rerunning the geometry calculation.

---
### MAC Simulator

The MAC simulator is implemented in `network/mac.py` using SimPy.

It supports:

- slotted CSMA/CA;
- unslotted CSMA/CA;
- periodic and stochastic packet generation;
- random initial node offsets;
- finite backoff and retransmission limits;
- CCA;
- Tx and ACK durations;
- ACK success/failure;
- collisions;
- hidden nodes;
- BTMA;
- physical uplink failure;
- physical downlink/ACK failure;
- per-node state times;
- multiple independent seeds.

#### Main entry points

##### Single simulation

```python
stats, params = run_sim(
    n_nodes=20,
    mean_iat_us=60e6,
    mode="unslotted",
    traffic_type="periodic",
    seed=1,
    sim_time_us=3000e6,
    data_rate_bps=10e3,
    symbol_rate_sym_s=10e3,
    payload_bytes=128,
    ack_bytes=16,
    hidden_node_mask=None,
    phy_pdr_up=None,
    phy_pdr_down=None,
    bt_hidden_mask=None,
    btma_mode=True,
    debug=False,
)
```

##### Multiple seeds

```python
mean, std, all_runs, per_node, params = call_MAC(
    nodes=20,
    period=60e6,
    mode="unslotted",
    traffic_type="periodic",
    n_seeds=150,
    sim_time_us=3000e6,
    data_rate_bps=10e3,
    symbol_rate_sym_s=10e3,
    payload_bytes=128,
    ack_bytes=16,
    hidden_node_mask=hidden_mask,
    bt_hidden_mask=bt_hidden_mask,
    btma_mode=True,
    phy_pdr_up=phy_pdr_up,
    phy_pdr_down=phy_pdr_down,
    log=True,
    debug=False,
)
```

##### Node-count sweep

```python
sweep = run_sweep(
    node_sweep=[20, 40, 60, 80, 100],
    mode="unslotted",
    n_seeds=150,
    base_seed=1,
    mean_iat_us=60e6,
    traffic_type="periodic",
    sim_time_us=3000e6,
    data_rate_bps=10e3,
    symbol_rate_sym_s=10e3,
    payload_bytes=128,
    ack_bytes=16,
    btma_mode=True,
)
```

#### `MAC_Params`

`MAC_Params` stores MAC timing and frame parameters. Its methods calculate:

- symbol duration;
- unit-backoff duration;
- CCA duration;
- SIFS duration;
- turnaround duration;
- complete frame length;
- frame duration;
- ACK duration.

The CSMA/CA limits, backoff exponents, and timing constants should be verified in `MAC_Params` for the selected standard and PHY.

#### Slotted mode

`SlotClock` implements a shared slot boundary. Nodes align their backoff and CCA operations to this clock.

#### Unslotted mode

In unslotted mode, backoff intervals are evaluated directly in simulation time without waiting for common slot boundaries.

#### Channel model

`VLC_Channel` represents the shared broadcast medium.

A hidden-node matrix follows:

```text
hidden_node_mask[tx, rx] = True
```

meaning that `rx` cannot hear `tx`.

The channel tracks:

- concurrent transmissions;
- whether nodes can sense one another;
- whether the AP can receive an uplink;
- collision status;
- active MAC service intervals;
- BTMA visibility.

#### Physical-link failures

The MAC simulation distinguishes:

1. collision losses;
2. uplink PHY losses;
3. downlink/ACK PHY losses;
4. channel-access failures.

#### BTMA

With BTMA enabled, the busy tone informs nodes about activity at the access point even when they cannot directly detect another sensor's IR transmission.

`bt_hidden_mask` identifies nodes that cannot detect the busy tone. Such nodes can remain hidden despite BTMA.

#### Node statistics

`NodeStats` records generated, delivered, and failed packets and accumulates timing and delay metrics.

The aggregation functions provide:

- packet delivery ratio;
- AP-received delivery ratio;
- collision counts;
- PHY uplink losses;
- PHY downlink/ACK losses;
- channel-access failures;
- retransmissions;
- CCA attempts;
- backoff slots;
- Tx attempts;
- mean delay;
- 99th-percentile delay;
- per-state times;
- channel and MAC active-time metrics.

Use `per_node_aggregate()` when energy consumption must be mapped back to individual sensors.

---
### Energy Model

`EnergyManager` combines PHY telemetry, hardware parameters, cycle tasks, and optional MAC output.

```python
energy = EnergyManager(
    phy_data=telemetry,
    design=design,
    config=config,
    MAC=True,
    btma_mode=True,
    MAC_mode="unslotted",
)
```

#### Cycle phases

The modeled cycle contains:

1. initialization;
2. sensing;
3. processing;
4. uplink Tx;
5. CCA;
6. turnaround/wait;
7. downlink Rx;
8. sleep.

Baseline durations are:

```text
t_sensing   = N_s_up / f_s
t_processing = N_c_up / f_mcu
t_tx         = L_up / Rb_up
t_rx         = L_dw / Rb_down
```

When MAC is enabled, Tx, CCA, Rx, wait, and related durations are replaced by per-node MAC results.

#### Receiver current

The receiver-current model distinguishes PD and PV nodes:

```text
PD Rx: MCU + ADC + TIA
PV Rx: MCU + ADC
```

This prevents TIA current from being assigned to a PV receiver.

#### Tx current

For IR sensors, the Tx current combines MCU current and the IR-driver current derived from optical Tx power.

For RF sensors, `RF_calc_I()` converts RF Tx power to current using the configured RF-driver model.

#### Cycle energy

Active energy is integrated as:

```text
E_active = V × sum(I_state × t_state)
```

Sleep energy is:

```text
E_sleep = V × I_sleep × max(0, T_cycle - t_active)
```

Total cycle energy is:

```text
E_cycle = E_active + E_sleep
```

#### Daily energy

Daily consumption is obtained from cycle energy and the configured cycle period. The manager stores:

```text
E_day_consumed
E_day_harvested
E_day_net
```

#### Harvesting

PV harvesting uses the active PV operating voltage and current exported by `PhyNet`, together with:

- harvesting hours;
- MPP efficiency;
- PV-node mask.

Non-PV nodes receive zero harvested energy.

#### Battery lifetime

Initial stored battery energy is derived from:

```text
capacity [mAh] × battery voltage × 3.6 × initial SoC
```

Battery lifetime depends on daily net energy. A node with non-negative net daily energy does not receive a finite depletion time in the idealized model. In practice, self-discharge, conversion losses, aging, and finite charge capacity should be added for long-term battery-state studies.

#### Results

Retrieve a per-node Pandas table using:

```python
df = energy.get_results_df()
```

Save it using:

```python
energy.save_csv("energy_results.csv")
```

The output contains node-level communication, energy, harvesting, and battery metrics. When MAC is enabled, MAC statistics are included where available.

---
### Performance Considerations

The principal computational costs are:

- wall discretization;
- high-order diffuse reflection;
- RIS element count;
- sensor-to-sensor channel matrices;
- large sensor populations;
- number of MAC seeds;
- MAC simulation duration.

Practical methods to reduce runtime include:

- use coarse wall grids during debugging;
- reduce diffuse bounces during model verification;
- disable RIS and windows when not required;
- enable BTMA to avoid optical sensor-to-sensor gain calculation when consistent with the corresponding paper;
- cache PHY results for repeated MAC runs or optimization problems with a static channel;
- run MAC seeds in parallel;
- separate budget runs from large parameter sweeps.

A full PHY calculation should not be repeated for every MAC seed when geometry and Tx powers are unchanged.

---
### Known Limitations and Implementation Notes

1. **Single-master alignment**  
   Automatic optical-uplink alignment with `budget_run=True` is implemented only for one master node.

2. **Binary PHY availability in energy-layer MAC integration**  
   The current energy workflow thresholds SNR to produce binary PHY success values. A smooth BER-to-PDR mapping would provide a more physical transition.

3. **Configuration schema is not formally validated**  
   The design dictionary relies on runtime parsing and array-shape checks.

4. **Idealized battery model**  
   Battery aging, self-discharge, maximum charge state, and nonlinear discharge behavior are not modeled.

5. **Diffuse-model cost**  
   High wall resolution and reflection order can produce large intermediate matrices.

6. **Mixed-network testing**  
    Mixed IR/RF and PD/PV populations are supported structurally, but every combination should be validated independently before publication-scale sweeps.

## Synopsis

A simulation is performed in four stages:

1. the design dictionary defines the room, nodes, optical surfaces, blockers, hardware, communication parameters, and energy profile;
2. `PhyNet` constructs the scenario and evaluates LoS, diffuse, RIS, RF, ambient-light, receiver-noise, SNR, and PHY-link quantities;
3. `PhyResultsDTO` transfers the required PHY outputs to the MAC and energy layers;
4. `EnergyManager` calculates cycle and daily energy, PV harvesting, battery lifetime, and, when `MAC=True`, runs the MAC simulator and replaces nominal communication durations with simulated per-node state times.

The main models are summarized as follows:

- optical LoS gain follows the Lambertian model with receiver FOV and blockage;
- diffuse gain is evaluated over discretized room surfaces;
- RIS gain is calculated over Tx--RIS--Rx paths;
- RF propagation follows the configured indoor path-loss model;
- spectral overlap combines the source spectrum, detector response, and optical-filter transmission;
- PD receivers include TIA and shot noise;
- PV receivers include the DC operating point, small-signal response, bandwidth, noise, and harvesting output;
- the MAC model includes CCA, backoff, collisions, retransmissions, hidden nodes, PHY failures, ACK failures, and BTMA;
- the energy model integrates the current and duration of initialization, sensing, processing, CCA, Tx, waiting, Rx, and sleep states.


## License
`pyenlight` is released as open-source software under the MIT License. This repository will accompany the submission of the corresponding paper, and the formal citation will be added upon publication. 
