import copy
import numpy as np
from dataclasses import dataclass, field, fields
import matplotlib.pyplot as plt
from typing import Dict, List

from pyenlight.hardware.physics import SpectralPhysics
from pyenlight.core.config import EnLightConfig
from pyenlight.core.utils import to_vec_Nx3, to_scal_Nx1
from pyenlight.core.config import EnLightConfig


@dataclass
class Elements:
    """Base data structure representing a batch of spatial elements in 3D space."""
    r: np.ndarray  # (N, 3) Position
    n: np.ndarray = field(default_factory=lambda: np.array([0,0,1]))  # (N, 3) Normal

    def __post_init__(self):
        self.r = np.array(self.r)
        self.n = np.array(self.n)

        if self.r.ndim == 1:
            self.r = self.r.reshape(1, 3)

        self.N = self.r.shape[0]
        self.n = to_vec_Nx3(self.N, self.n)
        norms = np.linalg.norm(self.n, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0):
            self.n = self.n / norms

    def __add__(self, other):
        if other is None:
            return copy.deepcopy(self)
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot add {type(other)} to {type(self)}")

        new_obj = copy.deepcopy(self)
        for f in fields(self):
            name = f.name
            val_self = getattr(self, name)
            val_other = getattr(other, name)

            if val_self is None and val_other is None:
                continue
            if val_self is None or val_other is None:
                raise ValueError(f"Field '{name}' is None in one object but not the other.")

            stacked_val = np.vstack([val_self, val_other])
            setattr(new_obj, name, stacked_val)

        new_obj.N = new_obj.r.shape[0]
        return new_obj

    @classmethod
    def merge(cls, batch_list):
        if not batch_list:
            return None

        merged = copy.deepcopy(batch_list[0])
        for f in fields(cls):
            name = f.name
            values = [getattr(b, name) for b in batch_list]

            if any(v is None for v in values):
                if all(v is None for v in values):
                    continue
                valid_sample = next(v for v in values if v is not None)
                cols = valid_sample.shape[1] if valid_sample.ndim > 1 else 1
                values = [
                    v if v is not None else np.zeros((b.N, cols))
                    for v, b in zip(values, batch_list)
                ]
            
            setattr(merged, name, np.vstack(values))

        merged.N = merged.r.shape[0]
        return merged



@dataclass
class OpticalRxElements(Elements):
    A: np.ndarray = None
    fov: np.ndarray = None
    refl: np.ndarray = None
    type_Rx: np.ndarray = None
    config: EnLightConfig = field(default_factory=EnLightConfig) 

    def __post_init__(self):
        super().__post_init__()
        
        # If values are missing, fall back to the config 
        if self.A is None: self.A = self.config.env.rx_area
        if self.fov is None: self.fov = self.config.env.fov
        if self.refl is None: self.refl = self.config.env.reflectivity
        if self.type_Rx is None: self.type_Rx = 0
        
        self.A = to_scal_Nx1(self.N, self.A)
        self.fov = to_scal_Nx1(self.N, self.fov)
        self.refl = to_scal_Nx1(self.N, self.refl)
        self.type_Rx = to_scal_Nx1(self.N, self.type_Rx)

@dataclass
class OpticalTxElements(Elements):
    p: np.ndarray = None
    m: np.ndarray = None
    config: EnLightConfig = field(default_factory=EnLightConfig) 

    def __post_init__(self):
        super().__post_init__()
        
        # Safety net for Transmitters
        if self.p is None: self.p = self.config.comm.VLC_Tx_power
        if self.m is None: self.m = self.config.env.m
        
        self.p = to_scal_Nx1(self.N, self.p)
        self.m = to_scal_Nx1(self.N, self.m)

@dataclass
class RFTxElements(Elements):
    p: np.ndarray = None #dBm

    def __post_init__(self):
        super().__post_init__()
        if self.p is None:
            raise ValueError("RFTxElements requires explicit 'p'.")
        self.p = to_scal_Nx1(self.N, self.p)

class Surface:
    def __init__(self, config: EnLightConfig, center, dims, const_axis, resolution, 
                 nT=None, nR=None, refl=None, type='Wall', name=None, P=None, sun_power=0.0):
        
        self.config = config
        self.r_surface, self.A = self.gen_surface_points(center, dims, const_axis, resolution)
        self.const_axis = const_axis
        self.name = name

        target_nT = nT if nT is not None else self.config.physics.zp 
        target_nR = nR if nR is not None else self.config.physics.zp 
        
        self.refl = refl if refl is not None else self.config.env.reflectivity
        self.type = type
        self.P = 0 if P is None else P

        if self.type == "window":
            
            sp = SpectralPhysics(self.config)
            
            
            spectral_integral = sp.sun_power()
            
            
            self.P = self.config.physics.pd_peak * self.A * spectral_integral
            print("power is "+ str(self.P))

        self.Tx_elements = OpticalTxElements(
            r=self.r_surface, 
            n=target_nT, 
            p=self.P, 
            m=self.config.env.m  
        ) 

        self.Rx_elements = OpticalRxElements(
            r=self.r_surface, 
            n=target_nR, 
            A=self.A, 
            refl=self.refl, 
            fov=self.config.env.fov,
            type_Rx=0 
        )

    @staticmethod
    def gen_surface_points(center, dims, const_axis, resolution):
        dim_1, dim_2 = dims
        res_1, res_2 = resolution

        grid_1 = np.linspace(-dim_1/2 + dim_1/res_1/2, dim_1/2 - dim_1/res_1/2, res_1)
        grid_2 = np.linspace(-dim_2/2 + dim_2/res_2/2, dim_2/2 - dim_2/res_2/2, res_2)
        mesh_1, mesh_2 = np.meshgrid(grid_1, grid_2)
        zeros = np.zeros_like(mesh_1)

        if const_axis == 0: offsets = np.stack([zeros, mesh_1, mesh_2], axis=-1)
        elif const_axis == 1: offsets = np.stack([mesh_1, zeros, mesh_2], axis=-1)
        else: offsets = np.stack([mesh_1, mesh_2, zeros], axis=-1)

        points = center + offsets.reshape(-1, 3)
        patch_area = float((dim_1/res_1)*(dim_2/res_2))
        return points, patch_area

class RoomBuilder:
    def __init__(self, design: Dict, config: EnLightConfig, console=False):
        self.design = design
        self.config = config
        self.env = self.design.get('environment', {})
        self.get_dimensions_and_res(console)
        self.get_reflectivity(console)
        self.get_surfaces_by_type("RIS", console)
        self.get_surfaces_by_type("window", console)
        self.blockers = self.env.get('blockers', None)

    def get_dimensions_and_res(self, console=False):
        dims = self.env.get('dimensions', self.config.env.room_dim)
        self.L, self.W, self.H = dims[0], dims[1], dims[2]
        self.res = self.env.get('wall_resolution', self.config.env.wall_resolution)
        if isinstance(self.res, int):
            self.res = (self.res, self.res)

    def get_reflectivity(self, console=False):
        refl_cfg = self.env.get('reflectivity', {})
        def_refl = self.config.env.reflectivity
        
        self.floor_refl = refl_cfg.get('floor', def_refl)
        self.ceiling_refl = refl_cfg.get('ceiling', def_refl)
        self.wall_refl = refl_cfg.get('walls', def_refl)
        self.refl = [self.floor_refl, self.ceiling_refl, self.wall_refl]

    def get_surfaces_by_type(self, surface_type, console=False):
        surfaces = self.env.get('special_surfaces', [])
        if surface_type == 'window':
            self.windows = [s for s in surfaces if s.get('type') == surface_type]
        elif surface_type == 'RIS':
            self.RIS = [s for s in surfaces if s.get('type') == surface_type]
        else:
            raise ValueError("Invalid surface type. Must be 'window' or 'RIS'.")

class Room:
    def __init__(self, rb: RoomBuilder, config: EnLightConfig, ignore_RIS=False, ignore_windows=False, console=False):
        self.config = config
        L, W, H = rb.L, rb.W, rb.H
        res = rb.res
        refl = rb.refl

        self.windows = []
        self.RIS = []
        self.Tx_RIS_elements = None
        self.Rx_RIS_elements = None
        self.Tx_windows_elements = None
        self.h_ww = None

        self.floor = Surface(self.config, np.array([L/2, W/2, 0]), (L, W), 2, res, 
                             nR=self.config.physics.zp, nT=self.config.physics.zp, refl=refl[0], name='Floor')
        self.ceiling = Surface(self.config, np.array([L/2, W/2, H]), (L, W), 2, res, 
                               nR=self.config.physics.zm, nT=self.config.physics.zm, refl=refl[1], name='Ceiling')
        self.west_wall = Surface(self.config, np.array([0, W/2, H/2]), (W, H), 0, res, 
                                 nR=self.config.physics.xp, nT=self.config.physics.xp, refl=refl[2], name='West Wall')
        self.east_wall = Surface(self.config, np.array([L, W/2, H/2]), (W, H), 0, res, 
                                 nR=self.config.physics.xm, nT=self.config.physics.xm, refl=refl[2], name='East Wall')
        self.south_wall = Surface(self.config, np.array([L/2, 0, H/2]), (L, H), 1, res, 
                                  nR=self.config.physics.yp, nT=self.config.physics.yp, refl=refl[2], name='South Wall')
        self.north_wall = Surface(self.config, np.array([L/2, W, H/2]), (L, H), 1, res, 
                                  nR=self.config.physics.ym, nT=self.config.physics.ym, refl=refl[2], name='North Wall')

        self.walls = [self.floor, self.ceiling, self.west_wall, self.east_wall, self.south_wall, self.north_wall]
        self._build_master_element()
        self.blockers = rb.blockers

        if not ignore_RIS:
            for ris in rb.RIS:
                ris_surface = Surface(self.config, ris['center'], ris['dims'], ris['const_axis'], ris['resolution'], 
                                      ris['normal'], ris['normal'], ris['reflectivity'], ris['type'], ris['name'])
                self.add_surface(ris_surface)
                
        if not ignore_windows:
            sp = SpectralPhysics(self.config)
            default_sun_power = sp.sun_power()
            for window in rb.windows:
                window_surface = Surface(self.config, window['center'], window['dims'],
                    window['const_axis'], window['resolution'],
                    window['normal'], window['normal'],
                    window['reflectivity'], window['type'], window['name'],
                    sun_power=window.get('sun_power', default_sun_power))
                self.add_surface(window_surface)

    def _build_master_element(self):
        Tx_wall_elements = [s.Tx_elements for s in self.walls]
        Rx_wall_elements = [s.Rx_elements for s in self.walls]
        self.Tx_wall_elements = OpticalTxElements.merge(Tx_wall_elements)
        self.Rx_wall_elements = OpticalRxElements.merge(Rx_wall_elements)

    def plot_surface_addition(self, new_surface, overlap_mask):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        wall_r = self.Rx_wall_elements.r
        kept_r = wall_r[~overlap_mask]
        removed_r = wall_r[overlap_mask]

        ax.scatter(kept_r[:, 0], kept_r[:, 1], kept_r[:, 2],
                   c='blue', alpha=0.05, s=2, label='Existing Wall (Kept)')

        if removed_r.shape[0] > 0:
            ax.scatter(removed_r[:, 0], removed_r[:, 1], removed_r[:, 2],
                       c='red', alpha=0.8, s=10, label='Wall Tiles Removed')

        new_r = new_surface.Tx_elements.r
        ax.scatter(new_r[:, 0], new_r[:, 1], new_r[:, 2],
                   c='green', alpha=0.9, s=15, marker='s', label=f'New {new_surface.name}')

        const_axis = new_surface.const_axis
        active_axes = [i for i in range(3) if i != const_axis]

        limits = {}
        for axis in active_axes:
            vals = new_r[:, axis]
            unique = np.unique(np.round(vals, 5))
            if len(unique) > 1:
                step = np.mean(np.diff(unique))
            else:
                step = np.sqrt(new_surface.Tx_elements.A[0,0])

            limits[axis] = (vals.min() - step/2, vals.max() + step/2)

        base_val = new_r[0, const_axis]

        a1, a2 = active_axes
        min1, max1 = limits[a1]
        min2, max2 = limits[a2]

        corners_2d = [
            (min1, min2),
            (max1, min2),
            (max1, max2),
            (min1, max2),
            (min1, min2) 
        ]

        x_line, y_line, z_line = [], [], []
        for (c1, c2) in corners_2d:
            pt = [0, 0, 0]
            pt[const_axis] = base_val
            pt[a1] = c1
            pt[a2] = c2

            x_line.append(pt[0])
            y_line.append(pt[1])
            z_line.append(pt[2])

        ax.plot(x_line, y_line, z_line, color='black', linewidth=3, label='Border')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"Adding '{new_surface.name}' to Room")
        ax.legend()

        max_range = np.array([self.floor.Tx_elements.r[:,0].max(),
                              self.floor.Tx_elements.r[:,1].max(),
                              self.ceiling.Tx_elements.r[:,2].max()]).max()
        ax.set_xlim(0, max_range)
        ax.set_ylim(0, max_range)
        ax.set_zlim(0, max_range)

        plt.show()

    def add_surface(self, new_surface):
        if new_surface.type == 'RIS':
            self.RIS.append(new_surface)
            if self.Tx_RIS_elements is None:
                self.Tx_RIS_elements = new_surface.Tx_elements
                self.Rx_RIS_elements = new_surface.Rx_elements
            else:
                self.Tx_RIS_elements = self.Tx_RIS_elements + new_surface.Tx_elements
                self.Rx_RIS_elements = self.Rx_RIS_elements + new_surface.Rx_elements

        elif new_surface.type == 'window':
            self.windows.append(new_surface)
            if self.Tx_windows_elements is None:
                self.Tx_windows_elements = new_surface.Tx_elements
            else:
                self.Tx_windows_elements = self.Tx_windows_elements + new_surface.Tx_elements

        new_r = new_surface.Tx_elements.r
        old_r = self.Rx_wall_elements.r
        const_axis = new_surface.const_axis
        const_val = new_r[0, const_axis]
        active_axes = [i for i in range(3) if i != const_axis]

        overlap_mask = np.abs(old_r[:, const_axis] - const_val) < 1e-4

        for axis in active_axes:
            vals = new_r[:, axis]
            unique_vals = np.unique(np.round(vals, 5))
            if len(unique_vals) > 1:
                step = np.mean(np.diff(unique_vals))
            else:
                step = np.sqrt(new_surface.Tx_elements.A[0, 0])

            min_edge = np.min(vals) - step/2
            max_edge = np.max(vals) + step/2
            buffer = 1e-5
            is_inside_dim = (old_r[:, axis] > min_edge + buffer) & (old_r[:, axis] < max_edge - buffer)
            overlap_mask = overlap_mask & is_inside_dim

        removed_count = np.sum(overlap_mask)
        if 0 and removed_count > 0:
            self.plot_surface_addition(new_surface, overlap_mask)

        self.Rx_wall_elements.refl[overlap_mask, 0] = 0
