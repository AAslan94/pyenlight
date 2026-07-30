"""
==========================================================================
IEEE 802.15.7 CAP CSMA/CA Simulation -- VLC IoT Networks
==========================================================================
Models both Slotted and Unslotted CSMA/CA for Visible Light Communication
(IEEE 802.15.7). Select the variant with mode='slotted' | 'unslotted'.

PHY-aware: phy_pdr_up / phy_pdr_down per-node binary flags gate delivery
and ACK reception independently of MAC-layer collision outcome.

Protocol differences
---------------------
Slotted   -- Backoff countdown in whole slot units via a shared SlotClock.
             Two-stage CCA: sample at slot start AND slot end; both must be
             idle. TX begins on a slot boundary. After a failed ACK the
             node re-aligns to the next slot boundary before the next retry.

Unslotted -- Backoff is a continuous real-time wait (k * unit_backoff_us).
             Single-stage CCA: one sample taken after the backoff wait.
             TX can begin at any instant. No slot-boundary re-alignment.

Metrics tracked
---------------
  payload_throughput_kbps / frame_throughput_kbps / throughput_kbps
  active_payload_throughput_kbps / active_frame_throughput_kbps
  active_time_s                  -- union wall-clock time where >=1 node is active
  node_active_time_s             -- sum of per-node active times, diagnostic only
  pdr_ap                         -- Application PDR: frame reached AP
  pdr_mac                        -- Strict MAC PDR: frame reached AP AND ACK received
  mean_delay_us
  mean_delay_unconditional
  p99_delay_us
  failure_rate                   -- any missing ACK, MAC + PHY
  collision_rate                 -- true MAC collisions only
  true_mac_collision_rate
  offered_load
  pkts_gen / pkts_del_ap / pkts_del_mac
  drop_no_access / drop_no_ack
  collisions / mac_collisions / phy_drops_up / phy_drops_down
  cca_per_period / bo_slots_per_period / avg_retries_per_pkt
  mean_time_idle_us / mean_time_cca_us / mean_time_tx_us / mean_time_rx_us
==========================================================================
"""

import concurrent.futures
import simpy
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

matplotlib.rcParams["text.usetex"] = False


# =============================================================================
# 1. PARAMETERS
# =============================================================================

@dataclass
class MAC_Params:
    # -- Mode -----------------------------------------------------------------
    mode: str = "slotted"   # "slotted" | "unslotted"

    # -- PHY PDR ---------------------------------------------------------------
    phy_pdr_up: Optional[np.ndarray] = None
    phy_pdr_down: Optional[np.ndarray] = None

    # -- PHY ------------------------------------------------------------------
    data_rate_bps: float = 150_000
    symbol_rate_sym_s: float = 150_000

    # -- MAC Timings (symbols) ------------------------------------------------
    unit_backoff_symbols: int = 20
    cca_symbols: int = 8
    sifs_symbols: int = 12
    turnaround_symbols: int = 12

    # -- CSMA/CA Algorithm ----------------------------------------------------
    min_be: int = 3
    max_be: int = 5
    max_backoffs: int = 4
    max_retries: int = 3

    # -- Frame Structure (bytes) ----------------------------------------------
    payload_bytes: int = 100
    mhr_bytes: int = 9
    mfr_bytes: int = 2
    ack_bytes: int = 5

    # -- Traffic --------------------------------------------------------------
    mean_iat_us: float = 1_000_000.0
    traffic_type: str = "periodic"

    # -- Simulation Control ---------------------------------------------------
    n_nodes: int = 10
    sim_time_us: float = 5_000_000_000.0
    seed: Optional[int] = 43
    debug: bool = False

    # -- Network-level active-time tracking -----------------------------------
    # Union wall-clock time where at least one node is inside MAC service.
    union_active_time_us: float = 0.0

    # -- Derived Quantities ---------------------------------------------------
    @property
    def symbol_duration_us(self) -> float:
        return (1.0 / self.symbol_rate_sym_s) * 1e6

    @property
    def unit_backoff_us(self) -> float:
        return self.unit_backoff_symbols * self.symbol_duration_us

    @property
    def cca_duration_us(self) -> float:
        return self.cca_symbols * self.symbol_duration_us

    @property
    def sifs_us(self) -> float:
        return self.sifs_symbols * self.symbol_duration_us

    @property
    def turnaround_us(self) -> float:
        return self.turnaround_symbols * self.symbol_duration_us

    @property
    def frame_bytes(self) -> int:
        return self.payload_bytes + self.mhr_bytes + self.mfr_bytes

    @property
    def frame_duration_us(self) -> float:
        return (self.frame_bytes * 8 / self.data_rate_bps) * 1e6

    @property
    def ack_duration_us(self) -> float:
        return (self.ack_bytes * 8 / self.data_rate_bps) * 1e6


# =============================================================================
# 2. SLOT CLOCK
# =============================================================================

class SlotClock:
    """
    Shared slot clock for slotted CSMA/CA.
    """

    def __init__(self, env: simpy.Environment, params: MAC_Params):
        self.env = env
        self.p = params
        self._tick = env.event()
        self.slot_n = 0
        env.process(self._run())

    def _run(self):
        while True:
            yield self.env.timeout(self.p.unit_backoff_us)
            self.slot_n += 1
            old = self._tick
            self._tick = self.env.event()
            old.succeed()

    def wait_slots(self, n: int):
        """Yield for exactly n slot boundaries."""
        for _ in range(n):
            yield self._tick


# =============================================================================
# 3. DEBUG LOGGER
# =============================================================================

def _log(env: simpy.Environment, node_id: int, msg: str, debug: bool = False):
    if not debug:
        return
    print(f"[t {env.now:>12.2f}] Node {node_id}: {msg}")


def _log_channel(env: simpy.Environment, msg: str, debug: bool = False):
    if not debug:
        return
    print(f"[t {env.now:>12.2f}] CHANNEL: {msg}")


# =============================================================================
# 4. VLC CHANNEL
# =============================================================================

class VLC_Channel:
    """
    Shared broadcast medium with optional hidden-node and BTMA support.

    hidden_node_mask[tx, rx] = True -> rx CANNOT hear tx.
    """

    def __init__(
    self,
    env: simpy.Environment,
    hidden_node_mask: Optional[np.ndarray] = None,
    bt_hidden_mask: Optional[np.ndarray] = None,
    btma_mode: bool = False,
    ap_reachable: Optional[np.ndarray] = None,
    debug: bool = False,
    ):
        self.env = env
        self._active_tx: Dict[int, float] = {}
        self._collided: set = set()
        self._hidden = hidden_node_mask
        self._bt_hidden = bt_hidden_mask
        self._btma_mode = btma_mode
        self._debug = debug

        if ap_reachable is None:
            self._ap_reachable = None
        else:
            self._ap_reachable = np.asarray(ap_reachable, dtype=bool).flatten()

        # Network-level union active time:
        # active if at least one node is inside MAC service for a packet.
        self._active_mac_nodes = set()
        self._active_interval_start_us = None
        self.union_active_time_us = 0.0

        if self._hidden is not None:
            assert self._hidden.dtype == bool, "mask must be bool"
            n = self._hidden.shape[0]
            assert self._hidden.shape == (n, n), "mask must be square"
            assert not np.diag(self._hidden).any(), "diagonal must be False"

    @property
    def busy(self) -> bool:
        return bool(self._active_tx)

    def can_hear(self, listener: int, transmitter: int) -> bool:
        """True if listener CAN hear transmitter."""
        if self._hidden is None:
            return True
        return not self._hidden[transmitter, listener]

    def reaches_ap(self, node_id: int) -> bool:
        """
        Return True when the node's UL signal reaches the AP.

        If no PHY reachability array is supplied, preserve the original
        collision-channel behaviour and assume every node reaches the AP.
        """
        if self._ap_reachable is None:
            return True

        return bool(self._ap_reachable[node_id])

    def cca(self, node_id: int) -> bool:
        """True = idle, False = busy."""
        if not self._active_tx:
            return True

        if self._btma_mode:
            if self._bt_hidden is not None and self._bt_hidden[node_id]:
                return True
            return False

        for tx_id in self._active_tx:
            if self.can_hear(node_id, tx_id):
                return False
        return True

    def mac_active_start(self, node_id: int):
        """
        Mark node as active in MAC service.

        The union active interval starts only when the first node becomes active.
        """
        if not self._active_mac_nodes:
            self._active_interval_start_us = self.env.now
        self._active_mac_nodes.add(node_id)

    def mac_active_end(self, node_id: int):
        """
        Mark node as no longer active in MAC service.

        The union active interval ends only when the last active node becomes inactive.
        """
        self._active_mac_nodes.discard(node_id)

        if not self._active_mac_nodes and self._active_interval_start_us is not None:
            self.union_active_time_us += self.env.now - self._active_interval_start_us
            self._active_interval_start_us = None

    def close_active_interval(self, end_time_us: float):
        """
        Close an active interval if simulation stops while nodes are still active.
        """
        if self._active_mac_nodes and self._active_interval_start_us is not None:
            self.union_active_time_us += end_time_us - self._active_interval_start_us
            self._active_interval_start_us = None
            self._active_mac_nodes.clear()

    def start_tx(self, node_id: int):
        """
        Start a transmission.

        Two overlapping transmissions collide at the AP only when both
        transmitters can reach the AP according to the UL PHY threshold.
        """

        if self._active_tx:
            for tx_id in list(self._active_tx):

                new_reaches_ap = self.reaches_ap(node_id)
                active_reaches_ap = self.reaches_ap(tx_id)

                if new_reaches_ap and active_reaches_ap:
                    self._collided.add(tx_id)
                    self._collided.add(node_id)

                    hidden_str = (
                        f"node {node_id} CANNOT hear node {tx_id} "
                        f"-- hidden-node collision"
                        if not self.can_hear(node_id, tx_id)
                        else
                        f"node {node_id} CAN hear node {tx_id} "
                        f"-- concurrent-transmission collision"
                        )

                    _log_channel(
                        self.env,
                        f"COLLISION -- Node {node_id} started TX while "
                        f"Node {tx_id} was already transmitting "
                        f"({hidden_str})",
                        self._debug,
                        )

                else:
                    blocked_nodes = []

                    if not active_reaches_ap:
                        blocked_nodes.append(str(tx_id))

                    if not new_reaches_ap:
                        blocked_nodes.append(str(node_id))

                    _log_channel(
                        self.env,
                        f"NO AP COLLISION -- Nodes {tx_id} and {node_id} "
                        f"overlap, but node(s) {', '.join(blocked_nodes)} "
                        f"do not reach the AP",
                        self._debug,
                        )

        self._active_tx[node_id] = self.env.now

        _log_channel(
            self.env,
            f"Node {node_id} started TX | "
            f"reaches_ap={self.reaches_ap(node_id)} | "
            f"active_tx={list(self._active_tx.keys())}",
            self._debug,
            )

    def end_tx(self, node_id: int) -> bool:
        """Returns True if transmission was successful, i.e. no collision."""
        success = node_id not in self._collided
        self._active_tx.pop(node_id, None)
        self._collided.discard(node_id)

        _log_channel(
            self.env,
            f"Node {node_id} ended TX -> "
            f"{'SUCCESS' if success else 'FAILED (collision)'} | "
            f"active_tx={list(self._active_tx.keys())}",
            self._debug,
        )

        return success


# =============================================================================
# 5. PER-NODE STATISTICS
# =============================================================================

@dataclass
class NodeStats:
    node_id: int

    # -- Packet counters ------------------------------------------------------
    pkts_generated: int = 0
    pkts_delivered_ap: int = 0
    pkts_delivered_mac: int = 0

    pkts_dropped_no_access: int = 0
    pkts_dropped_no_ack: int = 0

    # -- Attempt / retry counters ---------------------------------------------
    total_retries: int = 0
    tx_attempts: int = 0
    collisions_detected: int = 0

    # -- Failure breakdown ----------------------------------------------------
    mac_collisions: int = 0
    phy_drops_up: int = 0
    phy_drops_down: int = 0

    # -- CSMA/CA algorithm counters -------------------------------------------
    total_backoff_slots: int = 0
    cca_attempts: int = 0

    # -- Latency --------------------------------------------------------------
    delays_us: List[float] = field(default_factory=list)
    all_delays_us: List[float] = field(default_factory=list)

    # -- Time-in-state --------------------------------------------------------
    time_idle_us: float = 0.0
    time_cca_us: float = 0.0
    time_tx_us: float = 0.0
    time_rx_us: float = 0.0

    def log_delay(self, gen_time: float, now: float):
        self.delays_us.append(now - gen_time)

    @property
    def mean_delay_us(self) -> float:
        return float(np.mean(self.delays_us)) if self.delays_us else 0.0

    @property
    def p99_delay_us(self) -> float:
        return float(np.percentile(self.delays_us, 99)) if self.delays_us else 0.0


# =============================================================================
# 6. VLC NODE
# =============================================================================

class VLC_Node:
    """
    IEEE 802.15.7 node running slotted or unslotted CSMA/CA.
    """

    def __init__(
        self,
        env: simpy.Environment,
        node_id: int,
        channel: VLC_Channel,
        clock: Optional[SlotClock],
        params: MAC_Params,
        stats: NodeStats,
    ):
        self.env = env
        self.nid = node_id
        self.ch = channel
        self.clk = clock
        self.p = params
        self.s = stats
        self._debug = params.debug
        self._tx_lock = simpy.Resource(env, capacity=1)

        rng_seed = (params.seed * 10_000 + node_id) if params.seed is not None else None
        self._rng = random.Random(rng_seed)

        env.process(self._packet_source())

    def _charge(self, state: str, duration_us: float):
        """Accumulate wall-clock time in a given MAC state."""
        if state == "idle":
            self.s.time_idle_us += duration_us
        elif state == "cca":
            self.s.time_cca_us += duration_us
        elif state == "tx":
            self.s.time_tx_us += duration_us
        elif state == "rx":
            self.s.time_rx_us += duration_us

    def _packet_source(self):
        if self.p.traffic_type == "periodic":
            yield self.env.timeout(self._rng.uniform(0, self.p.mean_iat_us))

        while True:
            if self.p.traffic_type == "poisson":
                iat = self._rng.expovariate(1.0 / self.p.mean_iat_us)
            elif self.p.traffic_type == "periodic":
                iat = self.p.mean_iat_us
            else:
                raise ValueError(f"Unknown traffic_type: {self.p.traffic_type!r}")

            yield self.env.timeout(iat)
            self.s.pkts_generated += 1
            _log(
                self.env,
                self.nid,
                f"Packet #{self.s.pkts_generated} generated -> queuing",
                self._debug,
            )
            self.env.process(self._queue_packet(self.env.now))

    def _queue_packet(self, gen_time: float):
        with self._tx_lock.request() as req:
            yield req

            self.ch.mac_active_start(self.nid)
            try:
                yield self.env.process(self._csma_ca(gen_time))
            finally:
                self.ch.mac_active_end(self.nid)

    def _align_to_slot_boundary(self):
        elapsed_in_slot = self.env.now % self.p.unit_backoff_us
        if elapsed_in_slot > 1e-9:
            remainder = self.p.unit_backoff_us - elapsed_in_slot
            self._charge("idle", remainder)
            yield self.env.timeout(remainder)

    def _csma_ca(self, gen_time: float):
        p, s, nid = self.p, self.s, self.nid
        slotted = p.mode == "slotted"

        pkt_delivered_ap = False
        pkt_delivered_mac = False

        for _retry in range(p.max_retries + 1):
            if _retry > 0:
                s.total_retries += 1
                _log(self.env, nid, f"--- Retry #{_retry} ---", self._debug)

            nb = 0
            be = p.min_be
            channel_access = False

            while nb <= p.max_backoffs:
                k = self._rng.randint(0, (2 ** be) - 1)
                s.total_backoff_slots += k

                _log(
                    self.env,
                    nid,
                    f"Backoff: waiting {k} slots (BE={be}, NB={nb})",
                    self._debug,
                )

                if slotted:
                    self._charge("idle", k * p.unit_backoff_us)
                    yield from self.clk.wait_slots(k)
                else:
                    backoff_us = k * p.unit_backoff_us
                    self._charge("idle", backoff_us)
                    yield self.env.timeout(backoff_us)

                s.cca_attempts += 1

                if slotted:
                    idle_start = self.ch.cca(nid)
                    _log(
                        self.env,
                        nid,
                        f"CCA[start] -> {'IDLE' if idle_start else 'BUSY'}",
                        self._debug,
                    )
                    self._charge("cca", p.cca_duration_us)
                    yield self.env.timeout(p.cca_duration_us)
                    idle_end = self.ch.cca(nid)
                    _log(
                        self.env,
                        nid,
                        f"CCA[end] -> {'IDLE' if idle_end else 'BUSY'}",
                        self._debug,
                    )
                    cca_passed = idle_start and idle_end
                else:
                    self._charge("cca", p.cca_duration_us)
                    yield self.env.timeout(p.cca_duration_us)
                    cca_passed = self.ch.cca(nid)
                    _log(
                        self.env,
                        nid,
                        f"CCA -> {'IDLE' if cca_passed else 'BUSY'}",
                        self._debug,
                    )

                if cca_passed:
                    _log(self.env, nid, "CCA PASSED -> proceeding to TX", self._debug)
                    channel_access = True
                    break

                _log(
                    self.env,
                    nid,
                    f"CCA FAILED -> NB={nb + 1}, BE={min(be + 1, p.max_be)}",
                    self._debug,
                )

                if slotted:
                    slot_remainder_us = p.unit_backoff_us - p.cca_duration_us
                    self._charge("idle", slot_remainder_us)
                    yield self.env.timeout(slot_remainder_us)

                nb += 1
                be = min(be + 1, p.max_be)

            if not channel_access:
                _log(
                    self.env,
                    nid,
                    "MAX BACKOFFS reached -> packet DROPPED (no access)",
                    self._debug,
                )
                s.pkts_dropped_no_access += 1
                s.all_delays_us.append(self.env.now - gen_time)
                return

            self._charge("idle", p.turnaround_us)
            yield self.env.timeout(p.turnaround_us)

            s.tx_attempts += 1
            _log(
                self.env,
                nid,
                f"Starting frame TX ({p.frame_duration_us:.1f} us)",
                self._debug,
            )

            self.ch.start_tx(nid)
            self._charge("tx", p.frame_duration_us)
            yield self.env.timeout(p.frame_duration_us)

            tx_mac_success = self.ch.end_tx(nid)

            pdr_up = p.phy_pdr_up[nid] if p.phy_pdr_up is not None else 1.0
            pdr_down = p.phy_pdr_down[nid] if p.phy_pdr_down is not None else 1.0

            phy_up_success = pdr_up > 0.5
            phy_down_success = pdr_down > 0.5

            ap_received_data = tx_mac_success and phy_up_success
            ack_received = ap_received_data and phy_down_success

            if ap_received_data and not pkt_delivered_ap:
                s.pkts_delivered_ap += 1
                pkt_delivered_ap = True

            if ack_received and not pkt_delivered_mac:
                s.pkts_delivered_mac += 1
                pkt_delivered_mac = True

            self._charge("idle", p.sifs_us)
            yield self.env.timeout(p.sifs_us)

            self._charge("rx", p.ack_duration_us)
            yield self.env.timeout(p.ack_duration_us)

            if slotted:
                yield from self._align_to_slot_boundary()

            if ack_received:
                s.log_delay(gen_time, self.env.now)
                _log(self.env, nid, "ACK received -> packet DELIVERED", self._debug)
                return

            s.collisions_detected += 1

            if not tx_mac_success:
                s.mac_collisions += 1
                _log(self.env, nid, "No ACK (MAC collision) -> will retry", self._debug)
            elif not phy_up_success:
                s.phy_drops_up += 1
                _log(self.env, nid, "No ACK (uplink SNR too low) -> will retry", self._debug)
            elif not phy_down_success:
                s.phy_drops_down += 1
                _log(
                    self.env,
                    nid,
                    "No ACK (downlink ACK SNR too low) -> will retry",
                    self._debug,
                )

        _log(
            self.env,
            nid,
            "MAX RETRIES reached -> packet DROPPED (no ACK)",
            self._debug,
        )
        s.pkts_dropped_no_ack += 1
        s.all_delays_us.append(self.env.now - gen_time)


# =============================================================================
# 7. RUNNER + AGGREGATORS
# =============================================================================

def run_sim(
    n_nodes: int,
    mean_iat_us: float,
    mode: str = "slotted",
    traffic_type: str = "periodic",
    seed: Optional[int] = None,
    sim_time_us: float = 300_000_000.0,
    data_rate_bps: float = 10_000,
    symbol_rate_sym_s: float = 10_000,
    payload_bytes: int = 100,
    ack_bytes: int = 5,
    hidden_node_mask: Optional[np.ndarray] = None,
    phy_pdr_up: Optional[np.ndarray] = None,
    phy_pdr_down: Optional[np.ndarray] = None,
    bt_hidden_mask: Optional[np.ndarray] = None,
    btma_mode: bool = False,
    debug: bool = False,
) -> tuple:
    """
    Run one simulation episode and return (list[NodeStats], MAC_Params).
    """
    assert mode in ("slotted", "unslotted"), (
        f"mode must be 'slotted' or 'unslotted', got {mode!r}"
    )

    params = MAC_Params(
        mode=mode,
        n_nodes=n_nodes,
        mean_iat_us=mean_iat_us,
        traffic_type=traffic_type,
        sim_time_us=sim_time_us,
        seed=seed,
        data_rate_bps=data_rate_bps,
        symbol_rate_sym_s=symbol_rate_sym_s,
        payload_bytes=payload_bytes,
        ack_bytes=ack_bytes,
        phy_pdr_up=phy_pdr_up,
        phy_pdr_down=phy_pdr_down,
        debug=debug,
    )

    env = simpy.Environment()
    clock = SlotClock(env, params) if mode == "slotted" else None
    channel = VLC_Channel(
        env,
        hidden_node_mask=hidden_node_mask,
        bt_hidden_mask=bt_hidden_mask,
        btma_mode=btma_mode,
        ap_reachable=phy_pdr_up,
        debug=debug,
    )

    all_stats: List[NodeStats] = []
    for i in range(n_nodes):
        st = NodeStats(node_id=i)
        VLC_Node(env, i, channel, clock, params, st)
        all_stats.append(st)

    env.run(until=sim_time_us)

    channel.close_active_interval(sim_time_us)
    params.union_active_time_us = channel.union_active_time_us

    return all_stats, params


def aggregate(node_stats: List[NodeStats], params: MAC_Params) -> dict:
    """Aggregate all-node statistics for one simulation episode."""
    sim_s = params.sim_time_us * 1e-6

    pkts_gen = sum(s.pkts_generated for s in node_stats)
    pkts_del_ap = sum(s.pkts_delivered_ap for s in node_stats)
    pkts_del_mac = sum(s.pkts_delivered_mac for s in node_stats)
    drop_no_access = sum(s.pkts_dropped_no_access for s in node_stats)
    drop_no_ack = sum(s.pkts_dropped_no_ack for s in node_stats)
    collisions = sum(s.collisions_detected for s in node_stats)
    tx_attempts = sum(s.tx_attempts for s in node_stats)
    mac_collisions = sum(s.mac_collisions for s in node_stats)
    phy_drops_up = sum(s.phy_drops_up for s in node_stats)
    phy_drops_down = sum(s.phy_drops_down for s in node_stats)
    total_retries = sum(s.total_retries for s in node_stats)
    cca_attempts = sum(s.cca_attempts for s in node_stats)
    bo_slots = sum(s.total_backoff_slots for s in node_stats)

    all_delays = [d for s in node_stats for d in s.delays_us]
    all_delays_unconditional = [d for s in node_stats for d in s.all_delays_us]

    payload_throughput_kbps = (
        (pkts_del_mac * params.payload_bytes * 8) / max(sim_s, 1e-12) / 1e3
    )
    frame_throughput_kbps = (
        (pkts_del_mac * params.frame_bytes * 8) / max(sim_s, 1e-12) / 1e3
    )

    node_active_time_us = sum(
        s.time_idle_us + s.time_cca_us + s.time_tx_us + s.time_rx_us
        for s in node_stats
    )

    active_time_us = getattr(params, "union_active_time_us", node_active_time_us)
    active_time_s = active_time_us * 1e-6
    node_active_time_s = node_active_time_us * 1e-6

    active_payload_throughput_kbps = (
        (pkts_del_mac * params.payload_bytes * 8)
        / max(active_time_s, 1e-12)
        / 1e3
    )

    active_frame_throughput_kbps = (
        (pkts_del_mac * params.frame_bytes * 8)
        / max(active_time_s, 1e-12)
        / 1e3
    )

    pdr_ap = pkts_del_ap / max(pkts_gen, 1)
    pdr_mac = pkts_del_mac / max(pkts_gen, 1)

    mean_success_delay_us = float(np.mean(all_delays)) if all_delays else 0.0
    p99_success_delay_us = float(np.percentile(all_delays, 99)) if all_delays else 0.0
    mean_delay_unconditional = (
        float(np.mean(all_delays_unconditional)) if all_delays_unconditional else 0.0
    )

    failure_rate = collisions / max(tx_attempts, 1)
    true_mac_col_rate = mac_collisions / max(tx_attempts, 1)

    empirical_offered_load = (pkts_gen * params.frame_duration_us * 1e-6) / max(sim_s, 1e-12)

    cca_per_period = cca_attempts / max(pkts_gen, 1)
    bo_slots_per_period = bo_slots / max(pkts_gen, 1)
    avg_retries_per_pkt = total_retries / max(pkts_gen, 1)

    mean_time_idle_us = sum(s.time_idle_us for s in node_stats) / max(pkts_gen, 1)
    mean_time_cca_us = sum(s.time_cca_us for s in node_stats) / max(pkts_gen, 1)
    mean_time_tx_us = sum(s.time_tx_us for s in node_stats) / max(pkts_gen, 1)
    mean_time_rx_us = sum(s.time_rx_us for s in node_stats) / max(pkts_gen, 1)

    return dict(
        payload_throughput_kbps=payload_throughput_kbps,
        frame_throughput_kbps=frame_throughput_kbps,
        throughput_kbps=payload_throughput_kbps,

        active_payload_throughput_kbps=active_payload_throughput_kbps,
        active_frame_throughput_kbps=active_frame_throughput_kbps,
        active_time_s=active_time_s,
        node_active_time_s=node_active_time_s,

        pdr_ap=pdr_ap,
        pdr_mac=pdr_mac,
        pdr=pdr_mac,

        mean_delay_us=mean_success_delay_us,
        mean_delay_unconditional=mean_delay_unconditional,
        p99_delay_us=p99_success_delay_us,

        failure_rate=failure_rate,
        collision_rate=true_mac_col_rate,
        true_mac_collision_rate=true_mac_col_rate,

        offered_load=empirical_offered_load,

        pkts_gen=pkts_gen,
        pkts_del_ap=pkts_del_ap,
        pkts_del_mac=pkts_del_mac,

        drop_no_access=drop_no_access,
        drop_no_ack=drop_no_ack,
        collisions=collisions,

        cca_per_period=cca_per_period,
        bo_slots_per_period=bo_slots_per_period,

        mean_time_idle_us=mean_time_idle_us,
        mean_time_cca_us=mean_time_cca_us,
        mean_time_tx_us=mean_time_tx_us,
        mean_time_rx_us=mean_time_rx_us,

        mac_collisions=mac_collisions,
        phy_drops_up=phy_drops_up,
        phy_drops_down=phy_drops_down,

        avg_retries_per_pkt=avg_retries_per_pkt,
    )


def per_node_aggregate(
    all_seeds_stats: List[List[NodeStats]],
    params: MAC_Params,
) -> List[dict]:
    """
    Average per-node MAC statistics across independent seed runs.
    Timings are normalised to per-packet values.
    """
    n_nodes = len(all_seeds_stats[0])
    per_node = []

    for i in range(n_nodes):
        node_runs = [seed_stats[i] for seed_stats in all_seeds_stats]

        def _safe_per_pkt(attr, s):
            return getattr(s, attr) / max(s.pkts_generated, 1)

        per_node.append({
            "node_id": i,
            "pdr_ap": float(np.mean([
                s.pkts_delivered_ap / max(s.pkts_generated, 1)
                for s in node_runs
            ])),
            "pdr_mac": float(np.mean([
                s.pkts_delivered_mac / max(s.pkts_generated, 1)
                for s in node_runs
            ])),
            "pdr": float(np.mean([
                s.pkts_delivered_mac / max(s.pkts_generated, 1)
                for s in node_runs
            ])),
            "true_mac_collision_rate": float(np.mean([
                s.mac_collisions / max(s.tx_attempts, 1)
                for s in node_runs
            ])),
            "failure_rate": float(np.mean([
                s.collisions_detected / max(s.tx_attempts, 1)
                for s in node_runs
            ])),
            "collision_rate": float(np.mean([
                s.mac_collisions / max(s.tx_attempts, 1)
                for s in node_runs
            ])),
            "phy_error_rate_up": float(np.mean([
                s.phy_drops_up / max(s.tx_attempts, 1)
                for s in node_runs
            ])),
            "phy_error_rate_down": float(np.mean([
                s.phy_drops_down / max(s.tx_attempts, 1)
                for s in node_runs
            ])),
            "drop_no_access": float(np.mean([
                s.pkts_dropped_no_access / max(s.pkts_generated, 1)
                for s in node_runs
            ])),
            "drop_no_ack": float(np.mean([
                s.pkts_dropped_no_ack / max(s.pkts_generated, 1)
                for s in node_runs
            ])),
            "avg_retries": float(np.mean([
                s.total_retries / max(s.pkts_generated, 1)
                for s in node_runs
            ])),
            "mean_time_tx_us": float(np.mean([
                _safe_per_pkt("time_tx_us", s) for s in node_runs
            ])),
            "mean_time_rx_us": float(np.mean([
                _safe_per_pkt("time_rx_us", s) for s in node_runs
            ])),
            "mean_time_cca_us": float(np.mean([
                _safe_per_pkt("time_cca_us", s) for s in node_runs
            ])),
            "mean_time_idle_us": float(np.mean([
                _safe_per_pkt("time_idle_us", s) for s in node_runs
            ])),
            "mean_backoff_slots": float(np.mean([
                s.total_backoff_slots / max(s.pkts_generated, 1)
                for s in node_runs
            ])),
            "mean_cca_attempts": float(np.mean([
                s.cca_attempts / max(s.pkts_generated, 1)
                for s in node_runs
            ])),
        })

    return per_node


# =============================================================================
# 8. call_MAC
# =============================================================================

def call_MAC(
    nodes: int,
    period: float,
    mode: str = "unslotted",
    traffic_type: str = "periodic",
    n_seeds: int = 1,
    sim_time_us: float = 500_000_000.0,
    data_rate_bps: float = 10e3,
    symbol_rate_sym_s: float = 10e3,
    payload_bytes: int = 100,
    ack_bytes: int = 5,
    hidden_node_mask: Optional[np.ndarray] = None,
    bt_hidden_mask: Optional[np.ndarray] = None,
    btma_mode: bool = False,
    phy_pdr_up: Optional[np.ndarray] = None,
    phy_pdr_down: Optional[np.ndarray] = None,
    log: bool = True,
    debug: bool = False,
    max_workers: Optional[int] = None,
) -> dict:
    """
    Run n_seeds independent simulations and return mean/std/all/per_node/params.

    If max_workers == 1, runs sequentially. This is useful in Spyder/Jupyter.
    """
    mean_iat_us = int(period * 1e6)
    base_seed = 41

    if log:
        hidden_status = (
            "WITH BTMA (AP-to-Sensor) hidden nodes" if btma_mode else
            "WITH CSMA (Sensor-to-Sensor) hidden nodes" if hidden_node_mask is not None else
            "NO hidden nodes"
        )
        print(
            f"\n  [{mode.upper()}] Nodes={nodes} IAT={mean_iat_us / 1e6:g} s "
            f"sim={sim_time_us / 1e6:g} s traffic={traffic_type.capitalize()} "
            f"seeds={base_seed}..{base_seed + n_seeds - 1} ({hidden_status})"
        )

    if debug and n_seeds > 1:
        print("  [WARNING] debug=True with n_seeds>1 produces very large output.")

    seed_aggs = []
    all_seeds_raw = []
    last_params = None

    run_configs = []
    for s in range(base_seed, base_seed + n_seeds):
        run_configs.append(dict(
            n_nodes=nodes,
            mean_iat_us=mean_iat_us,
            mode=mode,
            traffic_type=traffic_type,
            seed=s,
            sim_time_us=sim_time_us,
            data_rate_bps=data_rate_bps,
            symbol_rate_sym_s=symbol_rate_sym_s,
            payload_bytes=payload_bytes,
            ack_bytes=ack_bytes,
            hidden_node_mask=hidden_node_mask,
            bt_hidden_mask=bt_hidden_mask,
            btma_mode=btma_mode,
            phy_pdr_up=phy_pdr_up,
            phy_pdr_down=phy_pdr_down,
            debug=debug,
        ))

    if max_workers == 1:
        completed = 0
        for kw in run_configs:
            stats, params = run_sim(**kw)
            last_params = params
            all_seeds_raw.append(stats)
            seed_aggs.append(aggregate(stats, params))
            completed += 1

            if log:
                print(
                    f"  [{completed:3d}/{n_seeds}] "
                    f"PDR_AP={seed_aggs[-1]['pdr_ap']:.3f} "
                    f"PDR_MAC={seed_aggs[-1]['pdr_mac']:.3f} "
                    f"col={seed_aggs[-1]['collision_rate']:.3f}",
                    end="\r",
                )
    else:
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_sim, **kw): kw["seed"] for kw in run_configs}

            for future in concurrent.futures.as_completed(futures):
                stats, params = future.result()
                last_params = params
                all_seeds_raw.append(stats)
                seed_aggs.append(aggregate(stats, params))
                completed += 1

                if log:
                    print(
                        f"  [{completed:3d}/{n_seeds}] "
                        f"PDR_AP={seed_aggs[-1]['pdr_ap']:.3f} "
                        f"PDR_MAC={seed_aggs[-1]['pdr_mac']:.3f} "
                        f"col={seed_aggs[-1]['collision_rate']:.3f}",
                        end="\r",
                    )

    if log:
        print()

    keys = seed_aggs[0].keys()
    return {
        "mean": {k: float(np.mean([a[k] for a in seed_aggs])) for k in keys},
        "std": {k: float(np.std([a[k] for a in seed_aggs])) for k in keys},
        "all": seed_aggs,
        "per_node": per_node_aggregate(all_seeds_raw, last_params),
        "params": last_params,
    }


# =============================================================================
# 9. MULTI-SEED SWEEP
# =============================================================================

def run_sweep(
    node_sweep: List[int],
    mode: str = "slotted",
    n_seeds: int = 10,
    base_seed: int = 0,
    mean_iat_us: float = 1_000_000.0,
    traffic_type: str = "periodic",
    sim_time_us: float = 5_000_000_000.0,
    data_rate_bps: float = 10_000,
    symbol_rate_sym_s: float = 10_000,
    payload_bytes: int = 100,
    ack_bytes: int = 5,
    hidden_node_mask: Optional[np.ndarray] = None,
    bt_hidden_mask: Optional[np.ndarray] = None,
    btma_mode: bool = False,
    phy_pdr_up: Optional[np.ndarray] = None,
    phy_pdr_down: Optional[np.ndarray] = None,
    debug: bool = False,
) -> dict:
    """Sweep over node counts, averaging each metric over n_seeds seeds."""
    total_runs = len(node_sweep) * n_seeds
    print(
        f"\n[Sweep/{mode}] N={node_sweep} IAT={mean_iat_us / 1e6:g} s "
        f"sim={sim_time_us / 1e6:g} s traffic={traffic_type.capitalize()} "
        f"seeds={base_seed}..{base_seed + n_seeds - 1} ({total_runs} runs)"
    )

    results = {}
    run_count = 0

    for n in node_sweep:
        seed_aggs = []
        for s in range(base_seed, base_seed + n_seeds):
            stats, params = run_sim(
                n_nodes=n,
                mean_iat_us=mean_iat_us,
                mode=mode,
                traffic_type=traffic_type,
                seed=s,
                sim_time_us=sim_time_us,
                data_rate_bps=data_rate_bps,
                symbol_rate_sym_s=symbol_rate_sym_s,
                payload_bytes=payload_bytes,
                ack_bytes=ack_bytes,
                hidden_node_mask=hidden_node_mask,
                bt_hidden_mask=bt_hidden_mask,
                btma_mode=btma_mode,
                phy_pdr_up=phy_pdr_up,
                phy_pdr_down=phy_pdr_down,
                debug=debug,
            )
            seed_aggs.append(aggregate(stats, params))
            run_count += 1
            print(f"  [{run_count:3d}/{total_runs}] N={n:4d} seed={s}", end="\r")

        keys = seed_aggs[0].keys()
        results[n] = {
            "mean": {k: float(np.mean([a[k] for a in seed_aggs])) for k in keys},
            "std": {k: float(np.std([a[k] for a in seed_aggs])) for k in keys},
            "all": seed_aggs,
        }

    print()
    return results


# =============================================================================
# 10. PLOTTING
# =============================================================================

def plot_sweep(
    sweep_results: dict,
    node_sweep: List[int],
    mode: str = "slotted",
    save_path: str = "vlc_sweep.png",
):
    """Six-panel figure: mean +/- 1 sigma over all seeds."""
    ns = node_sweep

    def m(k):
        return [sweep_results[n]["mean"][k] for n in ns]

    def s(k):
        return [sweep_results[n]["std"][k] for n in ns]

    panels = [
        ("bo_slots_per_period", "Avg Backoff Slots / Period", "#2196F3"),
        ("pdr_ap", "Application PDR (AP Rx)", "#9C27B0"),
        ("pdr_mac", "Strict MAC PDR (ACK)", "#4CAF50"),
        ("collision_rate", "MAC Collision Rate", "#F44336"),
        ("mean_delay_us", "Mean MAC Delay (us)", "#FF9800"),
        ("active_payload_throughput_kbps", "Active Payload Throughput (kbps)", "#9C27B0"),
    ]

    n_seeds = len(sweep_results[ns[0]]["all"])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"IEEE 802.15.7 {mode.capitalize()} CSMA/CA -- VLC IoT Simulation\n"
        f"mean +/- 1-sigma over {n_seeds} seeds",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (key, label, color) in zip(axes.flat, panels):
        ax.errorbar(
            ns,
            m(key),
            yerr=s(key),
            marker="o",
            color=color,
            linewidth=2,
            markersize=7,
            capsize=4,
            elinewidth=1.2,
            ecolor=color,
            alpha=0.9,
        )
        ax.set_xlabel("Number of Nodes", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_xticks(ns)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {save_path}")


# =============================================================================
# 11. MAIN TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  IEEE 802.15.7 CAP CSMA/CA -- VLC IoT Network Profiler")
    print("=" * 72)

    node_sweep = [10]
    n_seeds = 100
    base_seed = 42
    test_iat_us = 1_000_000.0
    test_sim_time_us = 400_000_000.0
    data_rate_bps = 5e3
    symbol_rate_sym_s = 5_000
    payload_bytes = 10
    ack_bytes = 5
    test_traffic = "periodic"
    ENABLE_PLOTTING = True

    common_kwargs = dict(
        node_sweep=node_sweep,
        n_seeds=n_seeds,
        base_seed=base_seed,
        mean_iat_us=test_iat_us,
        traffic_type=test_traffic,
        sim_time_us=test_sim_time_us,
        data_rate_bps=data_rate_bps,
        symbol_rate_sym_s=symbol_rate_sym_s,
        payload_bytes=payload_bytes,
        ack_bytes=ack_bytes,
        hidden_node_mask=None,
        bt_hidden_mask=None,
        btma_mode=False,
        phy_pdr_up=None,
        phy_pdr_down=None,
        debug=False,
    )

    slotted_res = run_sweep(mode="slotted", **common_kwargs)
    unslotted_res = run_sweep(mode="unslotted", **common_kwargs)

    print()
    for label, res in [("SLOTTED", slotted_res), ("UNSLOTTED", unslotted_res)]:
        for n in node_sweep:
            mu = res[n]["mean"]
            print(
                f"{label:<10} N={n:4d} "
                f"PDR_MAC={mu['pdr_mac']:.4f} "
                f"PayloadThr={mu['payload_throughput_kbps']:.4f} kbps "
                f"ActiveThr={mu['active_payload_throughput_kbps']:.4f} kbps "
                f"UnionActive={mu['active_time_s']:.4f} s"
            )

    if ENABLE_PLOTTING:
        plot_sweep(slotted_res, node_sweep, mode="slotted", save_path="vlc_sweep_slotted.png")
        plot_sweep(unslotted_res, node_sweep, mode="unslotted", save_path="vlc_sweep_unslotted.png")
