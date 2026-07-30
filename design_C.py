import numpy as np
from pyenlight.core.utils import generate_grid, diagonal_points, align_to 


def generate_focusing_normals(center, dims, resolution, const_axis, light_pos, target_pos):
    """
    Calculates the perfect geometric bisector normal for every sub-element.
    This guarantees the panel faces slightly 'up' to catch the ceiling light, 
    while angling perfectly to bounce it 'down' into the target shadow.
    """
    dim_1, dim_2 = dims
    res_1, res_2 = resolution

    # Recreate the exact spatial grid of the RIS sub-elements
    grid_1 = np.linspace(-dim_1/2 + dim_1/res_1/2, dim_1/2 - dim_1/res_1/2, res_1)
    grid_2 = np.linspace(-dim_2/2 + dim_2/res_2/2, dim_2/2 - dim_2/res_2/2, res_2)
    mesh_1, mesh_2 = np.meshgrid(grid_1, grid_2)
    zeros = np.zeros_like(mesh_1)

    if const_axis == 0: offsets = np.stack([zeros, mesh_1, mesh_2], axis=-1)
    elif const_axis == 1: offsets = np.stack([mesh_1, zeros, mesh_2], axis=-1)
    else: offsets = np.stack([mesh_1, mesh_2, zeros], axis=-1)

    r_patches = np.array(center) + offsets.reshape(-1, 3) 

    # 1. Vector pointing FROM patches TO the Ceiling Light (Incidence)
    v_in = np.array(light_pos) - r_patches
    v_in = v_in / np.linalg.norm(v_in, axis=1, keepdims=True)

    # 2. Vector pointing FROM patches TO the Floor Target (Reflection)
    v_out = np.array(target_pos) - r_patches
    v_out = v_out / np.linalg.norm(v_out, axis=1, keepdims=True)

    # 3. The Bisector: Perfectly splits the difference for each individual patch
    normals = v_in + v_out
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Return as a standard Python list to avoid JSON serialization crashes!
    return normals

# --- Calculate the focusing array for RIS 1 ---
ris1_focus_normals = generate_focusing_normals(
    center=[5.0, 1.5, 2],
    dims=[1.0, 1.0],
    resolution=[10, 10],
    const_axis=0,
    light_pos=[2.5, 2.5, 3.0],   # Master TX location
    target_pos=[4.8, 0.2, 0.0]   # Shadow target location
)

ris2_focus_normals = generate_focusing_normals(
    center=[5.0, 3, 2],
    dims=[1.0, 1.0],
    resolution=[10, 10],
    const_axis=0,
    light_pos=[2.5, 2.5, 3.0],   # Master TX location
    target_pos=[4.8, 3.8, 0.0]   # Shadow target location
)

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
                "center": [0, 2.5, 1.5], 
                "dims": [1, 1],
                "const_axis": 0,           # Constant on the X axis
                "resolution": [20, 20],
                "normal": [1, 0, 0],       # Pointing into the room
                "reflectivity": 0.05,
             },
            {
                "type": "RIS",
                "name": "Mirror 1 (Targets Blocker 1 Shadow)",
                "center": [5.0, 1.5, 2], # Placed high on the East Wall
                "dims": [1.0, 1.0],        # 1 square meter
                "const_axis": 0,           # Constant on the X axis
                "resolution": [10, 10],
                # calculated bisector to bounce light specifically to [4.5, 0.2, 0.0]
                "normal":ris1_focus_normals, 
                "reflectivity": 0.95
            },
            
            {
                "type": "RIS",
                "name": "Mirror 2 (Targets Blocker 2 Shadow)",
                "center": [5.0, 3, 2], # Placed high on the East Wall
                "dims": [1.0, 1.0],        # 1 square meter
                "const_axis": 0,           # Constant on the X axis
                "resolution": [10, 10],
                # calculated bisector to bounce light specifically to [4.5, 0.2, 0.0]
                "normal":ris2_focus_normals, 
                "reflectivity": 0.95
            },

        ],
        "blockers": {
            "positions": [
                [4,1,0],
                [3.5,3.5,0]
            ],
            "radius": 0.3,       
            "height":1.7    
        } 
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
            "rx_area": 1e-4,
            "m": 1,
            "FOV": np.pi/2,
            "rx_type": 0,       # 0 = Photodiode, 1 = Solar Panel (PV)
            "uplink_type": 0,   # 0 = Infrared (OW), 1 = RF
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
            "Rb_down": 10e3,  # Downlink Bitrate
            "n_sp_d": 0.4,    # Downlink spectral efficiency
            "Rb_up": 10e3,    # Uplink Bitrate (can also be array per sensor)
            "n_sp_u": 0.4     # Uplink spectral efficiency
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

}



