import numpy as np
from scipy.special import lambertw

from enlight_iot.core.config import EnLightConfig
from enlight_iot.core.utils import to_scal_Nx1

class TIA:
    """
    Models the frequency response and noise performance of a Transimpedance Amplifier (TIA).
    """
    def __init__(self, config: EnLightConfig, **kwargs):
        self.config = config
        d = self.config.devices.tia

        self.RF = kwargs.get('RF', d['RF'])
        self.Vn = kwargs.get('Vn', d['Vn'])
        self.In = kwargs.get('In', d['In'])
        self.fncV = kwargs.get('fncV', d['fncV'])
        self.fncI = kwargs.get('fncI', d['fncI'])
        # Pull temperature from the core physics config if not provided
        self.temperature = kwargs.get('temperature', self.config.physics.T)

    def CF(self, B):
        return 1 / (2 * np.pi * B * self.RF)   

    def ZF(self, f, B):
        f = f[None, :]
        B = B[:, None]
        CF = self.CF(B)
        return self.RF / (1 + 1j * 2 * np.pi * f * CF * self.RF)

    def RF_psd(self, f, B):
        return (4 * self.config.physics.kB * self.temperature / self.RF) * np.ones((B.size, f.size))

    def SV_psd(self, f, B):
        f = f[None, :]
        Z = self.ZF(f.squeeze(), B)
        return (self.Vn**2 + self.Vn**2 * self.fncV / f) / np.abs(Z)**2

    def SI_psd(self, f, B):
        f = f[None, :]
        return self.In**2 + self.In**2 * self.fncI / f

    def psd(self, f, B):
        return self.RF_psd(f, B) + self.SV_psd(f, B) + self.SI_psd(f, B)

    def calc_noise_power(self, B, Nf=1000, fmin=0.1):
        B = np.atleast_1d(B)
        x = np.linspace(0, 1, Nf)
        f = fmin + x * (B[:, None] - fmin)  

        psd_vals = self.psd(f[0], B)        
        psd_vals *= (f <= B[:, None])       

        return np.trapezoid(psd_vals, f, axis=1)

class IRdriver:
    """
    Models the electro-optical characteristics of an Infrared LED driver.
    """
    def __init__(self, config: EnLightConfig, **kwargs):
        d = config.devices.ir_driver
        
        self.imax = kwargs.get('imax', d['imax'])
        self.imin = kwargs.get('imin', d['imin'])
        self.pol = np.array(kwargs.get('pol', d['pol']))
        self.polinv = np.array(kwargs.get('polinv', d['polinv']))
        
        self.Pmax = np.polyval(self.pol, self.imax)
        self.Pmin = np.polyval(self.pol, self.imin)

    def calc_I(self, P):
        I = np.polyval(self.polinv, P)
        I = np.atleast_1d(I)
        P = np.atleast_1d(P)
        
        I[I >= self.imax] = np.inf
        I[P >= self.Pmax] = np.inf
        
        if I.size == 1: return I.item()
        return I

    def calc_P(self, I):
        return np.polyval(self.pol, I)

def RF_calc_I(P, config: EnLightConfig, **kwargs):
    """
    Estimates the power consumption current of the RF transmitter.
    """
    d = config.devices.rf_driver
    
    p_min = kwargs.get('p_min', d['p_min'])
    p_max = kwargs.get('p_max', d['p_max'])
    pol = kwargs.get('pol', d['pol'])

    P_bounded = np.array(P, copy=True)
    P_bounded[P_bounded < p_min] = p_min
    
    over_limit_mask = P_bounded > p_max
    I = np.atleast_1d(np.polyval(pol, P_bounded) * 1e-3)
    
    if np.any(over_limit_mask):
        I = I.astype(float) 
        I[over_limit_mask] = np.inf
        
    return I.item() if np.isscalar(P) else I

class PV:
    """
    Photovoltaic Cell Model.
    """
    def __init__(self, config: EnLightConfig, Gsignal, Gamb, unscaled=True, run=True, **kwargs):
        self.config = config
        
        self.Gamb = np.array(Gamb).reshape(-1, 1)
        self.Gsignal = np.array(Gsignal).reshape(-1, 1)
        self.no_pv = self.Gamb.shape[0]
        self.Gref = 1000.0
        

        self._init_params(kwargs)

        if unscaled:
            self._scale_params()

        self._calc_dc_bias()

        self.V = self.Voc * np.linspace(0, 1, 40).reshape(1, -1) 
        self.I = self.pv_current(self.V)
        self.I[self.I <= 0] = 1e-20  

        self.P = self.I * self.V
        self.ind = np.argmax(self.P, axis=1) 
        self.Pmax = np.take_along_axis(self.P, self.ind[:, None], axis=1)
        self.Rl = self.V / self.I

        self.ID = self.I0 * np.exp((self.V + self.I * self.Rs) / (self.n * self.Vt))
        self.r = (self.n * self.Vt) / self.ID
        self.iac = self.Iph * self.Gsignal/self.Gamb

        self.calc_capacitance() 
        self.find_bw()

        self.bw_ind = np.take_along_axis(self.BW, self.ind[:, None], axis=1) 
        
        f_steps = 60
        self.f = np.linspace(100, self.bw_ind.flatten(), f_steps).T

        if run:
            self.tf(self.f)
            self._thermal_noise_base()
            self.compute_all_noise(self.f)
            self.shot_noise(self.f)
            self.tf(self.f)
            self.vp2p(self.f)

    def _init_params(self, kwargs):
        d = self.config.devices.pv_circuit
        keys = ['A', 'n', 'Rs', 'Rsh', 'Voc', 'Jsc', 'Lo', 'Co', 'Rc', 
                'Na', 'Nd', 'L', 'er', 'ni']
        for key in keys:
            val = kwargs.get(key, d.get(key))
            setattr(self, key, to_scal_Nx1(self.no_pv, val))
        

    def _scale_params(self):
        scale = self.A * 1e4
        self.Rsh = self.Rsh / scale
        self.Rs = self.Rs / scale
        self.Isc = self.Jsc * scale
        

    def _calc_dc_bias(self):
        if not hasattr(self, 'Isc'): self.Isc = self.Jsc
        self.Iph = self.Isc.copy()
        
        # Pulled from core physics config
        self.Vt = self.config.physics.kB * self.config.physics.T / self.config.physics.q

        self.Iph *= (self.Gamb + self.Gsignal) / self.Gref
        self.Isc = self.Iph 
        self.Rsh *= self.Gref / (self.Gamb + self.Gsignal)
        self.Voc += self.Vt * np.log((self.Gamb + self.Gsignal) / self.Gref + 1e-20) 

        num = self.Isc - self.Voc / self.Rsh
        den = np.exp(self.Voc / (self.n * self.Vt)) - 1
        self.I0 = num / den

    def pv_current(self, V):
        V = np.asarray(V)
        Rs, Rsh = self.Rs, self.Rsh
        Iph, I0 = self.Iph, self.I0
        nVt = self.n * self.Vt

        R_sum = Rs + Rsh
        term_linear = (Rsh * (Iph + I0) - V) / R_sum
        common_factor = Rsh / (nVt * R_sum)
        exponent_term = np.exp(common_factor * (V + Rs * (Iph + I0)))
        pre_factor = I0 * Rs * common_factor

        theta = pre_factor * exponent_term
        w_val = lambertw(theta).real

        return term_linear - (nVt / Rs) * w_val

    def calc_capacitance(self):
        # Pulled from core physics config
        es = self.er * self.config.physics.eo
        q = self.config.physics.q

        no = self.ni**2 / self.Na
        vbi = self.Vt * np.log(self.Na * self.Nd / self.ni**2)

        denom = 2 * (self.Na + self.Nd) * (vbi - self.V + 1e-6)
        denom[denom < 0] = 1e-20
        c_dep = self.A * np.sqrt((q * es * self.Na * self.Nd) / denom)
        c_dif = self.A * q * self.L * no * np.exp(self.V / self.Vt) / self.Vt
        self.C = c_dep + c_dif


    def find_bw(self, verbose=False):
        r_eq_inv = 1/self.Rsh + 1/self.r + 1/(self.Rs + self.Rc)
        self.req = 1 / r_eq_inv
        self.BW = 1 / (2 * np.pi * self.req * self.C)
        if verbose: print(f"BW: {self.BW}")

    def tf(self, f):
        w = 2 * np.pi * f[:, None, :]
        r, C, Rl = self.r[..., None], self.C[..., None], self.Rl[..., None]

        Zp = 1 / (1/self.Rsh[..., None] + 1/r + 1j*w*C)
        Zdc = 1j*w*self.Lo[..., None] + Rl
        Zac = self.Rc[..., None] + 1/(1j*w*self.Co[..., None])

        Zout = 1 / (1/Zac + 1/Zdc) + self.Rs[..., None]
        h1 = Zp / (Zp + Zout)
        h2 = Zdc / (Zac + Zdc)

        self.hpv = np.abs(h1 * h2 * self.Rc[..., None])

    def _thermal_noise_base(self):
        # Pulled from core physics config
        kT = 4 * self.config.physics.kB * self.config.physics.T
        self.No_r = kT * self.r
        self.No_Rs = kT * self.Rs
        self.No_Rsh = kT * self.Rsh
        self.No_Rl = kT * self.Rl
        self.No_Rc = kT * self.Rc

    def compute_all_noise(self, f):
        self._thermal_noise_base()

        w = 2 * np.pi * f[:, None, :]
        r = self.r[..., None]
        C = self.C[..., None]
        Rl = self.Rl[..., None]
        Rs = self.Rs[..., None]
        Rsh = self.Rsh[..., None]
        Rc = self.Rc[..., None]

        Z_Co = 1 / (1j * w * self.Co[..., None])
        Z_Lo = 1j * w * self.Lo[..., None]
        Z_Comm = Rc + Z_Co          
        Z_EH = Rl + Z_Lo            

        J_p = 1/r + 1j*w*C + 1/Rsh
        Z_source = Rs + (1/J_p)
        Z_sp = 1 / (1/Z_source + 1/Z_EH)
        den_rc = Z_Comm + Z_sp
        self.n_rc = np.abs(Rc / den_rc)**2 * self.No_Rc[..., None]

        r2s = 1 / (1/Z_EH + 1/Z_Comm)
        u1_rs = r2s / (Rs + (1/J_p) + r2s)
        u2_rs = Rc / Z_Comm
        self.n_rs = np.abs(u1_rs * u2_rs)**2 * self.No_Rs[..., None]

        r1l = (1/J_p) + Rs
        r2l = Z_Comm
        r3l = 1 / (1/r1l + 1/r2l)
        u1_rl = Rc / Z_Comm
        u2_rl = r3l / (Rl + r3l + Z_Lo)
        self.n_rl = np.abs(u1_rl * u2_rl)**2 * self.No_Rl[..., None]

        h1 = 1 / Z_Comm
        h2 = 1 / Z_EH
        z_load_eq = 1 / (h1 + h2) 

        denom_rsh = 1/(Rs + z_load_eq) + 1/r + 1j*w*C
        z_node_rsh = 1 / denom_rsh

        u1_rsh = z_node_rsh / (Rsh + z_node_rsh)
        u2_rsh = z_load_eq / (z_load_eq + Rs)
        u3_rsh = Rc / Z_Comm

        self.n_rsh = np.abs(u1_rsh * u2_rsh * u3_rsh)**2 * self.No_Rsh[..., None]

        denom_r = 1/(Rs + z_load_eq) + 1/Rsh + 1j*w*C
        z_node_r = 1 / denom_r

        u1_r = z_node_r / (r + z_node_r)
        u2_r = z_load_eq / (z_load_eq + Rs)
        u3_r = Rc / Z_Comm

        self.n_r = np.abs(u1_r * u2_r * u3_r)**2 * self.No_r[..., None]

        self.int_rc = np.trapezoid(self.n_rc, f[:, None, :], axis=2)
        self.int_rs = np.trapezoid(self.n_rs, f[:, None, :], axis=2)
        self.int_rl = np.trapezoid(self.n_rl, f[:, None, :], axis=2)
        self.int_rsh = np.trapezoid(self.n_rsh, f[:, None, :], axis=2)
        self.int_r = np.trapezoid(self.n_r, f[:, None, :], axis=2)

        self.th_noise = (self.int_rc + self.int_rs + self.int_rl +
                         self.int_rsh + self.int_r)


    def shot_noise(self, f):
        t = np.abs(self.hpv)**2
        integral = np.trapezoid(t, f[:, None, :], axis=2)
        # Pulled from core physics config
        self.sh_noise = 2 * self.config.physics.q * self.Iph * integral


    def vp2p(self, f):
        self.vac = self.hpv * self.iac[:, None]

