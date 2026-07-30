# enlight_iot/__init__.py

# ── The Core Stack (For running the master simulation) ──
from .core.config import EnLightConfig
from .network.orchestrator import PhyNet, oPhyGains
from .hardware.energy import EnergyManager

# ── Environment & Spatial ──
from .environment.geometry import Room, Surface, Elements, OpticalRxElements, OpticalTxElements, RFTxElements
from .environment.channel import ChannelEngine

# ── Hardware & Physics ──
from .hardware.devices import TIA, PV, IRdriver
from .hardware.physics import SpectralPhysics

# ── Network Components ──
from .network.nodes import NodeBuilder, SNManager, MNManager, ANManager

# ── MAC Layer Tools ──
from .network.mac import call_MAC, run_sweep, plot_sweep

__version__ = "1.0.0"
