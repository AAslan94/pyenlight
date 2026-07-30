import numpy as np
from pyenlight.core.utils import generate_grid

design = {

    # ==================================================
    # Environment
    # ==================================================

    "environment": {

        # Room dimensions [length, width, height] in m
        "dimensions": [5.0, 5.0, 3.0],

        # Number of patches used along each surface axis
        "wall_resolution": [20, 20],

        # Surface reflectivities
        "reflectivity": {
            "floor": 0.2,
            "ceiling": 0.6,
            "walls": 0.8,
        },

        # Optional windows and RIS surfaces
        "special_surfaces": [

            {
                # Supported values: "window" or "RIS"
                "type": "window",

                # User-defined surface name
                "name": "West Window",

                # Surface centre [x, y, z] in m
                "center": [0.0, 2.5, 1.5],

                # Surface dimensions in m
                "dims": [1.0, 1.0],

                # Constant coordinate:
                # 0 -> x, 1 -> y, 2 -> z
                "const_axis": 0,

                # Surface discretisation
                "resolution": [20, 20],

                # Surface normal
                "normal": [1.0, 0.0, 0.0],

                # Surface reflectivity
                "reflectivity": 0.05,

            },

            {
                "type": "RIS",
                "name": "East RIS",
                "center": [5.0, 2.5, 1.5],
                "dims": [1.0, 1.0],
                "const_axis": 0,
                "resolution": [20, 20],
                "normal": [-1.0, 0.0, 0.0],
                "reflectivity": 0.9,
            },
        ],

        # Optional human blockers
        "blockers": {

            # Blocker floor-plane centres [x, y, z]
            "positions": np.array([
                [2.5, 2.5, 0.0],
                [3.5, 2.0, 0.0],
            ]),

            # Blocker radius in m.
            # May be a scalar or a compatible array.
            "radius": 0.3,

            # Blocker height in m.
            # May be a scalar or a compatible array.
            "height": 1.7,
        },
    },


    # ==================================================
    # Nodes
    # ==================================================

    "nodes": {

        # ----------------------------------------------
        # Master nodes
        # ----------------------------------------------

        "masters": {

            # Required master-node positions
            "positions": np.array([
                [2.5, 2.5, 3.0],
            ]),

            # Optical transmitter normal
            "nT": [0.0, 0.0, -1.0],

            # Optical receiver normal
            "nR": [0.0, 0.0, -1.0],

            # Optical receiver area in m^2
            "rx_area": 1e-4,

            # Lambertian order of the VLC transmitter
            "m": 1,

            # Optical receiver FOV in rad
            "FOV": np.pi / 2,

            # VLC downlink optical transmit power in W
            "tx_power": 1.0,

            # RF receiver sensitivity in dBm
            "sensitivity": -100.0,

            # True enables the IR optical filter at the
            # master-node receiver
            "IR_pass_filter": True,
        },


        # ----------------------------------------------
        # Sensor nodes
        # ----------------------------------------------

        "sensors": {

            # Required sensor-node positions
            "positions": generate_grid(
                0.2, 4.8,
                0.2, 4.8,
                0.0,
                10, 10,
                False,
            ),

            # IR transmitter normal
            "nT": [0.0, 0.0, 1.0],

            # Downlink optical receiver normal
            "nR": [0.0, 0.0, 1.0],

            # Optical receiver area in m^2
            "rx_area": 1e-4,

            # Lambertian order of the IR transmitter
            "m": 1,

            # Optical receiver FOV in rad
            "FOV": np.pi / 2,

            # Receiver type:
            # 0 -> PD
            # 1 -> PV
            #
            # May be a scalar or one value per sensor.
            "rx_type": 0,

            # Uplink type:
            # 0 -> IR
            # 1 -> RF
            #
            # May be a scalar or one value per sensor.
            "uplink_type": 0,

            # IR uplink optical transmit power in W
            "IR_tx_power": 15e-3,

            # RF uplink transmit power in dBm
            "RF_tx_power": -20.0,

            # RF receiver sensitivity used for
            # sensor-to-sensor RF sensing
            "sensitivity": -100.0,

            # True enables the VLC optical filter at
            # PD-based sensor receivers
            "VLC_pass_filter": True,
        },


        # ----------------------------------------------
        # Optional ambient optical sources
        # ----------------------------------------------

        "ambient_nodes": {

            # Required ambient-source positions
            "positions": np.array([
                [1.25, 1.25, 3.0],
                [3.75, 1.25, 3.0],
                [1.25, 3.75, 3.0],
                [3.75, 3.75, 3.0],
            ]),

            # Ambient-source transmitter normals
            "nT": [0.0, 0.0, -1.0],

            # Lambertian order
            "m": 3,

            # Optical transmit power in W
            "tx_power": 5.0,
        },
    },


    # ==================================================
    # Communication parameters
    # ==================================================

    "energy_profile": {

        # Sensor-node operation period in s
        "T_cycle": 60.0,

        # ----------------------------------------------
        # Communication rates and bandwidth factors
        # ----------------------------------------------

        "communication": {

            # VLC downlink bit rate in bit/s
            "Rb_down": 10e3,

            # Downlink bandwidth factor
            "n_sp_d": 0.4,

            # Uplink bit rate in bit/s.
            #
            # May be a scalar or one value per sensor.
            # If omitted, IR and RF rates are selected
            # from EnLightConfig according to uplink_type.
            "Rb_up": 10e3,

            # Uplink bandwidth factor.
            #
            # May be a scalar or one value per sensor.
            "n_sp_u": 0.4,
        },


        # ----------------------------------------------
        # Hardware parameters
        # ----------------------------------------------

        "hardware": {

            # MCU clock frequency in Hz
            "f_mcu": 16e6,

            # Sensing frequency in Hz
            "f_s": 1e3,

            # Sensor-node operating voltage in V
            "voltage": 3.3,
        },


        # ----------------------------------------------
        # Task parameters
        # ----------------------------------------------

        "tasks": {

            # Number of acquired samples per cycle
            "N_s_up": 100,

            # Number of processing cycles per cycle
            "N_c_up": 1e3,

            # Uplink packet length in bit
            "L_up_bits": 1024,

            # Downlink or ACK packet length in bit
            "L_dw_bits": 128,
        },


        # ----------------------------------------------
        # State currents
        # ----------------------------------------------

        # MCU active current in A
        "I_mcu": 2.73e-3,

        # ADC current in A
        "I_adc": 0.7e-3,

        # External sensor current in A
        "I_ext": 1.0e-3,

        # Sleep current in A
        "I_sleep": 2e-6,

        # Wake-up current in A
        "I_wake": 1e-3,

        # TIA current in A
        "I_tia": 0.7e-3,


        # ----------------------------------------------
        # Battery parameters
        # ----------------------------------------------

        "battery": {

            # Battery capacity in mAh
            "battery_capacity_mAh": 500,

            # Nominal battery voltage in V
            "V_batt": 3.6,

            # Initial state of charge
            "initial_soc": 1.0,
        },


        # ----------------------------------------------
        # IR transmitter-current model
        # ----------------------------------------------

        "IRDriver": {

            # Maximum driver current in A
            "imax": 100e-3,

            # Minimum driver current in A
            "imin": 0.0,

            # Polynomial mapping current to optical power
            "pol": np.array([
                1.353e-1,
                1.868e-1,
                -1.017e-4,
            ]),

            # Inverse polynomial mapping optical power
            # to driver current
            "polinv": np.array([
                -1.740e1,
                5.329,
                5.618e-4,
            ]),
        },


        # ----------------------------------------------
        # RF transmitter-current model
        # ----------------------------------------------

        "RFDriver": {

            # Minimum supported RF transmit power in dBm
            "p_min": -20.0,

            # Maximum supported RF transmit power in dBm
            "p_max": 5.0,

            # Polynomial mapping RF power in dBm to
            # current consumption in mA
            "pol": np.array([
                0.24,
                8.8,
            ]),
        },


        # ----------------------------------------------
        # MAC simulation parameters
        # ----------------------------------------------

        "MAC": {

            # MAC simulation duration in us
            "sim_time_us": 3000e6,

            # Number of independent random seeds
            "n_seeds": 150,

            # SNR threshold used for optical links in dB
            "SNR_THRESHOLD_dB": 8.5,

            # Busy-tone detection threshold in dB
            "BUSY_TONE_THRESHOLD_dB": 8.5,

            # Print aggregate MAC output
            "log": True,

            # Print detailed event-level output
            "debug": False,
        },
    },


    # ==================================================
    # Protocol parameters
    # ==================================================

    "protocol": {

        # Initialisation duration in s
        "t_init": 5e-3,

        # Turnaround or waiting duration in s
        "t_wait": 1e-3,

        # Daily harvesting duration in h
        #
        # May be a scalar or values compatible with the
        # number of PV-equipped sensor nodes.
        "harvesting_hours": 5.0,
    },


    # ==================================================
    # Maximum-power-point conversion
    # ==================================================

    "MPP": {

        # Energy-conversion efficiency
        "mpp_eff": 0.8,
    },


    # ==================================================
    # Transimpedance amplifier
    # ==================================================

    "TIA": {

        # Feedback resistance in ohm
        "RF": 1e6,

        # Input-referred voltage-noise density
        "Vn": 15e-9,

        # Input-referred current-noise density
        "In": 400e-15,

        # Voltage-noise corner frequency in Hz
        "fncV": 1e3,

        # Current-noise corner frequency in Hz
        "fncI": 1e3,

        # TIA temperature in K
        "temperature": 300.0,
    },


    # ==================================================
    # PV equivalent-circuit parameters
    # ==================================================

    "PV_circuit": {

        # PV active area in m^2
        "A": 1e-4,

        # Diode ideality factor
        "n": 1.6,

        # Series resistance in ohm
        "Rs": 1.0,

        # Shunt resistance in ohm
        "Rsh": 1000.0,

        # Open-circuit voltage under the reference
        # condition in V
        "Voc": 0.64,

        # Short-circuit current density under the
        # reference condition in A/cm^2
        "Jsc": 35e-3,

        # Output inductor in H
        "Lo": 1e-6,

        # Communication coupling capacitor in F
        "Co": 1e-6,

        # Communication load resistance in ohm
        "Rc": 10.0,

        # Acceptor concentration
        "Na": 1.0e22,

        # Donor concentration
        "Nd": 1.0e25,

        # Diffusion length in m
        "L": 300e-6,

        # Relative permittivity
        "er": 11.68,

        # Intrinsic carrier concentration
        "ni": 1.0e16,
    },
}
