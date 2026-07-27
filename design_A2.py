import numpy as np
from enlight_iot.core.utils import generate_grid, diagonal_points, align_to 

bw_a1 = np.load("bw_a1.npy")
u_b = bw_a1.reshape(-1,)*0.4

master_design_example = {
    "environment": {
        "dimensions": [5.0, 5.0, 3.0], # L, W, H in meters
        "wall_resolution": [20, 20],   # Grid patches for diffuse reflections
        "reflectivity": {
            "floor": 0.2,
            "ceiling": 0.6,
            "walls": 0.8
        },
        "special_surfaces": [
            {
                "type": "window",
                "name": "South Window",
                "center": [0, 2.5, 1.5], # On the south wall (y=0)
                "dims": [1, 1],
                "const_axis": 0,           # Constant on the Y axis
                "resolution": [20, 20],
                "normal": [1, 0, 0],       # Pointing into the room
                "reflectivity": 0.05,
             },
            #{
            #    "type": "RIS",
            #    "name": "Wall Reflector",
            #    "center": [2.5, 5, 1.5], # On the east wall (x=L)
            #    "dims": [1.0, 1.0],
            #    "const_axis": 1,           # Constant on the X axis
            #    "resolution": [5, 5],
            #    "normal": [0, -1, 0],      # Pointing into the room (-y)
            #    "reflectivity": 0.95
            #},
            #{
            #    "type": "RIS",
            #    "name": "Wall Reflector2",
            #    "center": [5, 4.5, 1.5], # On the east wall (x=L)
            #    "dims": [1.0, 2.0],
            #    "const_axis": 0,           # Constant on the X axis
            #    "resolution": [5, 5],
            #    "normal": [-1,0, 0],      # Pointing into the room (-y)
            #    "reflectivity": 0.95
            #}
        ],
        #"blockers": {
            #"positions": [
            #    [1,3,0],
            #    [3,1,0]
            #],
            #"radius": 0.3,        # (Optional) Cylinder radius in meters. Defaults to 0.3m.
            #"height":1.7   # (Optional) Cylinder height in meters. Defaults to 1.7m.
        #} 
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
            "positions": generate_grid(0.2,4.8,0.2,4.8,0,60,60,False), 
            "nT": [0, 0, 1],    # TX pointing up
            #"nR": align_to(generate_grid(0.2,4.8,0.2,4.8,0,40,40,False),np.array([2.5,2.5,3])),      # RX pointing up
            "nR": np.array([0,0,1]),
            "rx_area": 10e-4,
            "m": 1,
            "FOV": np.pi/2,
            "rx_type": 1,       # 0 = Photodiode, 1 = Solar Panel (PV)
            "uplink_type": 1,   # 0 = Infrared (OW), 1 = RF
            "IR_tx_power": 15e-3, # 15 mW
            "RF_tx_power": -20.0, # dBm
            "sensitivity": -100,
            "VLC_pass_filter": False
        },
       # "ambient_nodes": {
       #     "positions": np.array([[1.0, 2.5, 3.0], [4.0, 2.5, 3.0]]), # Interfering lamps
       #     "nT": [0, 0, -1],
       #     "nR": [0, 0, -1],
          
        #}
    },
    
    "energy_profile": {
        "T_cycle": 60,
        "communication": {
            "Rb_down": u_b,  # Downlink Bitrate
            "n_sp_d": 0.4,    # Downlink Filter span
            "Rb_up": 250e3,    # Uplink Bitrate (can also be array per sensor)
            "n_sp_u": 0.4     # Uplink Filter span
        }
    },
    
    "TIA": {
        "RF": 1e6,
        "Vn": 15e-9,
        "In": 400e-15,
        "fncV": 1e3,
        "fncI": 1e3,
        "temperature": 300.0
    },
    
    #"PV_circuit": {
    #    "n": 1.6,
    #    "Rs": 1.0,
    #    "Rsh": 1000.0,
    #    "Voc": 0.64,
    #    "Jsc": 35e-3
    #}
}




