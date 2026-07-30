import numpy as np
from typing import Callable
from pyenlight.core.config import EnLightConfig

class SpectralPhysics:
    """
    Physics engine for calculating effective responsivity by integrating spectral 
    overlaps. Dynamically uses the provided EnLightConfig for all physical constants.
    """
    
    def __init__(self, config: EnLightConfig):
        self.config = config
        
        # Pre-define the spectrum configurations for easy lookup
        self.CONFIGURATIONS = {
            "WLED2PD":   (self.white_led_spectrum, self.photodiode_responsivity, "ALL_PASS"),
            "WLED2PDwF": (self.white_led_spectrum, self.photodiode_responsivity, "VLC_PASS"),
            "IR2PD":     (self.tsff5210_spectrum, self.photodiode_responsivity, "ALL_PASS"),
            "IR2PDwF":   (self.tsff5210_spectrum, self.photodiode_responsivity, "IR_PASS"),
            "WLED2PV":   (self.white_led_spectrum, self.solar_panel_sensitivity, "ALL_PASS"),
            "SUN2PDwFv": (self.sun_spectrum, self.photodiode_responsivity, "VLC_PASS"),
            "SUN2PDwFi": (self.sun_spectrum, self.photodiode_responsivity, "IR_PASS"),
            "SUN2PD":    (self.sun_spectrum, self.photodiode_responsivity, "ALL_PASS"),
            "SUN2PV":    (self.sun_spectrum, self.solar_panel_sensitivity, "ALL_PASS"),
            "IR2PV":     (self.tsff5210_spectrum, self.solar_panel_sensitivity, "ALL_PASS"),
            "IR2PDwX":   (self.tsff5210_spectrum, self.photodiode_responsivity, "VLC_PASS")
        }

    # --- Helper Methods ---
    @staticmethod
    def _gaussian(wl: np.ndarray, peak: float, fwhm: float) -> np.ndarray:
        """Generates a Gaussian curve based on Peak and FWHM."""
        sigma = fwhm / (2 * np.sqrt(np.log(2)))
        return np.exp(-(wl - peak)**2.0 / sigma**2.0)

    @staticmethod
    def _poly_response(wl: np.ndarray, coeffs: np.ndarray, l_min_nm: float, l_max_nm: float) -> np.ndarray:
        """Evaluates a polynomial response within a specific nanometer range."""
        wl_nm = wl / 1e-9
        response = np.zeros_like(wl)
        mask = (wl_nm >= l_min_nm) & (wl_nm <= l_max_nm)
        if np.any(mask):
            x_scaled = 2 * wl_nm[mask] / (l_min_nm + l_max_nm)
            response[mask] = np.polyval(coeffs, x_scaled)
        return response

    # --- Source Definitions ---
    def white_led_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Models a White LED spectrum as a sum of two Gaussians."""
        blue = self._gaussian(wl, peak=470e-9, fwhm=20e-9)
        phosphor = self._gaussian(wl, peak=600e-9, fwhm=100e-9)
        return blue + phosphor

    def tsff5210_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Models the TSFF5210 IR Emitter spectrum (Peak ~870nm)."""
        return self._gaussian(wl, peak=870e-9, fwhm=40e-9)

    def sun_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Approximates Solar Spectrum using the config object."""
        T = self.config.spectral.t_sun
        lmax = self.config.physics.bK / T
        
        def blackbody(lam):
            num = 2 * self.config.physics.hP * self.config.physics.c0**2
            den = lam**5 * (np.exp((self.config.physics.hP * self.config.physics.c0) / 
                                   (lam * self.config.physics.kB * T)) - 1)
            return num / den
            
        Pmax = blackbody(lmax)
        return blackbody(wl) / Pmax

    # --- Detector Definitions ---
    def photodiode_responsivity(self, wl: np.ndarray) -> np.ndarray:
        """Spectral responsivity of the system photodiode."""
        coeffs = np.array([-6.39503882, 27.47316339, -45.57791267,
                           36.01964536, -12.8418451, 1.73076976])
        return self._poly_response(wl, coeffs, 330, 1090)

    def solar_panel_sensitivity(self, wl: np.ndarray) -> np.ndarray:
        """Spectral sensitivity of the system Solar Panel."""
        coeffs = np.array([26.78555644, -160.24353775, 381.86564712, -463.07816469,
                           300.12488471, -97.25192023, 12.34949208])
        return self._poly_response(wl, coeffs, 300, 1175)

    # --- Core Physics Logic ---
    def sun_power(self) -> float:
        """Calculates total integrated power using config limits."""
        wl = np.linspace(self.config.spectral.l_min, self.config.spectral.l_max, self.config.spectral.grid_points)
        xd = self.sun_spectrum(wl)
        return float(np.trapezoid(xd, wl))


    @staticmethod
    def get_filter_transmission(name: str, wl: np.ndarray) -> np.ndarray:
        """Returns transmission window using legacy pyowiot Super-Gaussian filters."""
        if name == "VLC_PASS":
            lpeak = (320e-9 + 720e-9) / 2
            l10 = 320e-9
            m = 6
            B = (lpeak - l10) / ( -np.log(0.1) ) ** (1/m)
            return np.exp( -(wl - lpeak)**m / B**m )
            
        elif name == "IR_PASS":
            lpeak = 900e-9
            l10 = 750e-9
            m = 6
            B = (lpeak - l10) / ( -np.log(0.1) ) ** (1/m)
            return np.exp( -(wl - lpeak)**m / B**m )
            
        return np.ones_like(wl)

    def calculate_effective_responsivity(self, source_func: Callable,
                                         detector_func: Callable,
                                         filter_name: str = "ALL_PASS") -> float:
        """Calculates Source * Detector * Filter overlap integral."""
        wl = np.linspace(self.config.spectral.l_min, self.config.spectral.l_max, self.config.spectral.grid_points)
        P_src = source_func(wl)
        R_det = detector_func(wl)
        T_filt = self.get_filter_transmission(filter_name, wl)
        
        total_power = np.trapezoid(P_src, wl)
        if total_power == 0: return 0.0
        
        detected_power = np.trapezoid(P_src * R_det * T_filt, wl)
        return float(detected_power / total_power)

    def get_responsivity_by_name(self, config_name: str) -> float:
        """Retrieves and calculates responsivity by configuration name."""
        if config_name not in self.CONFIGURATIONS:
            raise ValueError(f"Unknown config: {config_name}")
            
        src_func, det_func, filt_n = self.CONFIGURATIONS[config_name]
        return self.calculate_effective_responsivity(src_func, det_func, filt_n)
