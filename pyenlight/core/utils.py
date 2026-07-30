import numpy as np
from scipy.special import erfc, erfcinv

def to_vec_Nx3(N: int, val) -> np.ndarray:
    """
    Broadcasting helper: Ensures the input 'val' is shaped as (N, 3).
    Useful for expanding a single position/normal vector to match a batch of N elements.
    """
    arr = np.array(val)
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
    if arr.shape[0] == 1 and N > 1:
        arr = np.tile(arr, (N, 1))
    return arr

def to_scal_Nx1(N: int, val, default_val=0) -> np.ndarray:
    """
    Broadcasting helper: Ensures the input 'val' is shaped as (N, 1).
    Useful for expanding a scalar property (like Power or Area) to match a batch of N elements.
    """
    if val is None:
        val = default_val

    if is_scalar(val):
        return np.full((N, 1), val)

    arr = np.array(val)
    if arr.ndim == 1:
        return arr.reshape(N, 1)
    return arr

def normalize_bool_array(x, N: int) -> np.ndarray:
    """Converts a boolean scalar or list into a boolean numpy array of size N."""
    if isinstance(x, (bool, np.bool_)):
        return np.full(N, x, dtype=bool)
    return np.asarray(x, dtype=bool)

def as_array_of_size(x, N: int) -> np.ndarray:
    """Enforces that input 'x' becomes an array of size N."""
    if np.isscalar(x):
        return np.full(N, x)
    arr = np.asarray(x)
    if arr.size == 1:
        return np.full(N, arr.item())
    
    if arr.size != N:
        raise ValueError(f"Expected size {N}, got {arr.size}")
    return arr

def is_scalar(x) -> bool:
    """Checks if a value is effectively a scalar (int, float, or 0-d array)."""
    if x is None:
        return False
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
    
def solar_panel_angular_efficiency(cos_inc: np.ndarray) -> np.ndarray:
    """
    Models the angular loss of a Solar Panel (deviations from Lambertian).
    
    Calculates efficiency degradation based on the incidence angle (theta) using 
    a 5th-order polynomial fit derived from experimental data.
    
    Args:
        cos_inc (np.ndarray): Cosine of the incidence angle.
        
    Returns:
        np.ndarray: Efficiency scaling factor (0.0 to ~1.0).
    """
    p_p = np.array([-1.81907071e-09,  3.00750020e-07, -1.82841164e-05,  4.57546496e-04,
                    -4.11754977e-03,  1.00666212e+00])
    p_s = np.poly1d(p_p)
    theta = np.rad2deg(np.arccos(np.clip(cos_inc, -1.0, 1.0)))
    efficiency = p_s(theta)
    efficiency[theta >= 90] = 0
    return np.maximum(0, efficiency)

def calculate_blockage_mask(tx_r, rx_r, blocker_pos, r_h=0.3, h_h=1.7):
    """
    Vectorized human cylinder blockage model.
    Checks if a cylinder of radius r_h and height h_h intersects the LoS path.
    
    Uses full chord intersection (not just closest point) for exact cylinder
    intersection detection.
    """
    if blocker_pos is None or len(blocker_pos) == 0:
        return np.zeros((tx_r.shape[0], rx_r.shape[0]), dtype=bool)
    
    # Expand dimensions for broadcasting
    tx_r = tx_r[:, None, :]  # (Ntx, 1, 3)
    rx_r = rx_r[None, :, :]  # (1, Nrx, 3)
    bp = np.array(blocker_pos).reshape(-1, 1, 1, 3)  # (Nb, 1, 1, 3)
    
    # Vector from TX to RX
    d = rx_r - tx_r  # (Ntx, Nrx, 3)
    d_xy = d[..., :2]  # (Ntx, Nrx, 2)
    ray_len_2d = np.linalg.norm(d_xy, axis=2) + 1e-12  # (Ntx, Nrx)
    
    # Vector from TX to Blocker (XY only)
    v_xy = bp[..., :2] - tx_r[..., :2]  # (Nb, Ntx, Nrx, 2)
    
    # Unit direction in XY plane
    u_xy = d_xy / ray_len_2d[..., None]  # (Ntx, Nrx, 2)
    
    # 1. Perpendicular Distance Check
    # Distance from blocker axis to the ray line in XY plane
    # Using cross product: |v × u| where u is unit direction
    cross_2d = np.abs(v_xy[..., 0] * u_xy[..., 1] - v_xy[..., 1] * u_xy[..., 0])
    dist_to_line = cross_2d  # u is unit, so no need to divide
    is_crossing = dist_to_line < r_h
    
    # 2. Longitudinal Boundary Check
    dot_prod = np.sum(v_xy * u_xy, axis=-1)  # (Nb, Ntx, Nrx)
    d_along_ray = dot_prod  # u is unit, so projection length
    is_between = (d_along_ray + r_h > 0) & (d_along_ray - r_h < ray_len_2d)
    
    # 3. Height Check - Full chord intersection
    # Calculate the half-length of the chord where the ray is inside the cylinder footprint
    chord_half_len = np.sqrt(np.maximum(0, r_h**2 - dist_to_line**2))
    
    # Entry and exit parameters along the 2D ray
    d_entry = np.clip(d_along_ray - chord_half_len, 0, ray_len_2d)
    d_exit = np.clip(d_along_ray + chord_half_len, 0, ray_len_2d)
    
    # Convert to t parameters (0 at TX, 1 at RX)
    t_entry = d_entry / (ray_len_2d + 1e-12)
    t_exit = d_exit / (ray_len_2d + 1e-12)
    
    # Get z-values at entry and exit points
    z_tx = tx_r[..., 2]  # (Ntx, 1) -> broadcasts to (Nb, Ntx, Nrx)
    z_rx = rx_r[..., 2]  # (1, Nrx) -> broadcasts to (Nb, Ntx, Nrx)
    
    # Need to broadcast z_tx and z_rx to match (Nb, Ntx, Nrx)
    z_tx_b = z_tx[None, :, :]  # (1, Ntx, Nrx) -> broadcasts to (Nb, Ntx, Nrx)
    z_rx_b = z_rx[None, :, :]  # (1, Ntx, Nrx) -> broadcasts to (Nb, Ntx, Nrx)
    
    z_entry = z_tx_b + t_entry * (z_rx_b - z_tx_b)
    z_exit = z_tx_b + t_exit * (z_rx_b - z_tx_b)
    
    # The z-range of the ray while inside the horizontal footprint
    z_min_seg = np.minimum(z_entry, z_exit)
    z_max_seg = np.maximum(z_entry, z_exit)
    
    # Human cylinder occupies Z-space from 0 to h_h
    is_hitting_body = (z_min_seg < h_h) & (z_max_seg >= 0)
    
    # 4. Vertical Ray Check (Tx and Rx are vertically aligned, ray_len_2d ≈ 0)
    is_vertical = ray_len_2d < 1e-6
    dist_to_A_2d = np.linalg.norm(v_xy, axis=-1)  # distance from blocker to TX in XY
    
    z_min_ray = np.minimum(z_tx_b, z_rx_b)
    z_max_ray = np.maximum(z_tx_b, z_rx_b)
    z_overlap = (z_min_ray < h_h) & (z_max_ray >= 0)
    
    vertical_hit = is_vertical & (dist_to_A_2d < r_h) & z_overlap
    
    # Combine everything
    # For non-vertical rays: must pass all 3 checks
    blocked_normal = ~is_vertical & is_crossing & is_between & is_hitting_body
    blocked = blocked_normal | vertical_hit
    
    # Reduce over blockers: any blocker blocks the path
    # blocked shape: (Nb, Ntx, Nrx) -> reduce axis 0 to get (Ntx, Nrx)
    total_blockage = np.any(blocked, axis=0)
    
    return total_blockage

def generate_microgrids_vectorized(r, n, A, k=3):
    """Generates kxk sub-tiles for soft blockage calculations."""
    N = r.shape[0]
    
    # Calculate step size based on total patch Area
    step = np.sqrt(A.flatten()) / k 
    offsets = np.linspace(-(k-1)/2, (k-1)/2, k)
    
    # Create a local 2D grid
    x_off, y_off = np.meshgrid(offsets, offsets)
    grid_2d = np.stack([x_off.flatten(), y_off.flatten()], axis=1) # (k*k, 2)
    
    # Expand array to hold the k*k points for each of the N patches
    microgrids = np.repeat(r[:, np.newaxis, :], k*k, axis=1) # (N, k*k, 3)
    
    # Map the 2D grid onto the 3D plane using the normal vector
    for i in range(N):
        normal = np.abs(n[i])
        # Find which axes are parallel to the wall (where normal is 0)
        axes = np.where(normal < 0.5)[0] 
        if len(axes) == 2:
            microgrids[i, :, axes[0]] += grid_2d[:, 0] * step[i]
            microgrids[i, :, axes[1]] += grid_2d[:, 1] * step[i]

    return microgrids   

def Qfunction(x):
    """
    Standard Gaussian Q-function.
    Q(x) = 0.5 * erfc(x / sqrt(2))
    """
    return 0.5 * erfc( x/np.sqrt(2) )

def Qinv(y):
    """
    Inverse Gaussian Q-function.
    """
    return np.sqrt(2) * erfcinv( 2 * y )

def generate_grid(x_start,x_end,y_start,y_end,z_height, num_points_x, num_points_y,plot = False):
    x = np.linspace(x_start, x_end, num_points_x)
    y = np.linspace(y_start, y_end, num_points_y)
    xx, yy = np.meshgrid(x, y)
    z = np.full_like(xx, z_height)
    ceiling_points = np.stack([xx, yy, z], axis=-1).reshape(-1, 3)
    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(ceiling_points[:, 0], ceiling_points[:, 1], color='blue', marker='o', label='Lighting LEDs')
        plt.scatter(5,5,color = 'red', marker = 'x', label='Communication LED')  
        plt.title("LEDs Arrangement in the ceiling")
        plt.xlabel("Width [m]")
        plt.ylabel("Length [m]")
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim([0,10])
        plt.ylim([0,10])
        plt.legend(loc='upper right')
        plt.show()        
    return ceiling_points

def diagonal_points(x_start,x_fin,y_start,y_fin,height, N = 20):
	start = np.array([x_start, y_start,height])
	end = np.array([x_fin,y_fin,height])
	return np.linspace(start, end, N+2)[1:-1]

def align_to(r_rec, r_tra):
    """
    Calculates the unit vector direction from the receiver's position (r_rec) 
    to the transmitter's position (r_tra).

    Parameters:
    r_rec (np.ndarray): Receiver's position(s). Can be a 1D vector (N,) 
                        or a 2D array of vectors (M, N).
    r_tra (np.ndarray): Transmitter's position. Must be a 1D vector (N,).

    Returns:
    np.ndarray: The normalized unit vector(s) representing the alignment direction.
    """
    # 1. Calculate the displacement vector: (Transmitter - Receiver)
    # This vector points from r_rec to r_tra.
    # Broadcasting handles the subtraction for single or multiple receiver positions.
    displacement = r_tra - r_rec

    # 2. Calculate the magnitude of the displacement vectors
    # axis=-1 ensures the norm is calculated along the vector components (the last axis).
    # keepdims=True ensures the norm can be correctly broadcast back for division.
    norm = np.linalg.norm(displacement, axis=-1, keepdims=True)

    # Handle the zero-length vector case to prevent division by zero
    # np.where is used for a safe division.
    # It returns [0., 0., 0.] for any vector with a magnitude of 0.
    unit_vector = np.where(norm == 0, 0, displacement / norm)

    return unit_vector
