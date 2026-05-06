"""
points.py — Point data structures and set bookkeeping.

A Point is the represantation of a unique 3D point. 
These  carry across frames: stereo pixel coordinates (current and previous), 3D position/velocity/
covariance estimates from the previous and current solves, plus
identity and age.

A PointSet groups points by their role in the algorithm:
  F      : feature points in the EKF state (inertial references)
  F_pre  : candidate features awaiting two-frame solve & admission
  I      : interest points, steered by focus

Set membership defines role; points themselves are role-agnostic.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional
import cv2 as cv
import numpy as np



class PixelType(Enum):
    """Availability of stereo pixel observations across (k-1, k).

    Each slot ∈ {N, M, S}:
        N = no observation
        M = mono (one side; check which slot is populated)
        S = stereo (both sides)

    Post-push, the *_N states represent the interval before tracking
    or stereo extraction has populated the current frame's slots.
    """
    N_N   = "N-N"        # nothing in either frame; transient between
                         # construction and first observation
    N_M   = "N-M"        # newborn mono
    N_S   = "N-S"        # newborn stereo
    M_N   = "M-N"        # post-push, mono prev, awaiting current
    M_M   = "M-M"        # tracked mono throughout
    M_S   = "M-S"        # mono prev, recovered stereo this frame
    S_N   = "S-N"        # post-push, stereo prev, awaiting current
    S_M   = "S-M"        # lost a side this frame
    S_S   = "S-S"        # ideal: stereo throughout


class StateType(Enum):
    """Availability of 3D state estimates across (k-1, k).

    Each slot ∈ {N, P, PV}:
        N  = no estimate
        P  = position only
        PV = position + velocity

    """
    N_N    = "N-N"       # newborn, no solve has run yet
    N_P    = "N-P"       # first solve, position only
    N_PV   = "N-PV"      # impossible: no prev → no velocity.
    P_N    = "P-N"       # post-push, position-only prev, awaiting solve
    P_P    = "P-P"       # position carried; still no velocity
    P_PV   = "P-PV"      # velocity established this frame
    PV_N   = "PV-N"      # post-push, mature prev, awaiting solve
    PV_P   = "PV-P"      # regression: had velocity, now don't.
                         # Possible after partial-data solve. Flag for review.
    PV_PV  = "PV-PV"     # mature throughout


# =============================================================================
# Point — single 3D point
# =============================================================================

@dataclass
class Point:
    """One tracked point across two consecutive pair of frames."""

    # ---- identity --------------------------------------------------------
    id: int                                    # globally unique
    n: int = 0                                 # age in frames since birth

    # ---- pixel observations (stereo, current and previous frame) ---------
    # cv.KeyPoint carries (pt, size, angle, response, octave, class_id).
    # We exploit pt for the (u, v); other fields are detector metadata.
    uL_prev: Optional[cv.KeyPoint] = None      # u^{k-1}, left
    uR_prev: Optional[cv.KeyPoint] = None      # u^{k-1}, right
    uL_curr: Optional[cv.KeyPoint] = None      # u^{k},   left
    uR_curr: Optional[cv.KeyPoint] = None      # u^{k},   right

    # ---- 3D state from the previous solve (in B_{k-1}) -------------------
    p_prev:     Optional[np.ndarray] = None    # (3,)   position
    v_prev:     Optional[np.ndarray] = None    # (3,)   velocity
    Sigma_prev: Optional[np.ndarray] = None    # (3, 3), cov [p] if v is none but p populated, else (6, 6) joint cov [p; v]

    # ---- 3D state from the current solve (in B_k) ------------------------
    p_curr:     Optional[np.ndarray] = None
    v_curr:     Optional[np.ndarray] = None
    Sigma_curr: Optional[np.ndarray] = None

    # =====================================================================
    # API
    # =====================================================================

    def roll(self) -> None:
        """Roll k → k-1: current frame's quantities become previous, and
        previous quantities are discarded. Age increments by one. 
        """
        ...

    def get_px_type(self) -> PixelType:
        """Classify the point by which stereo pixel observations are
        populated in the previous and current frame.
        """
        prev = self._slot_kind(self.uL_prev, self.uR_prev)
        curr = self._slot_kind(self.uL_curr, self.uR_curr)
        return PixelType(f"{prev}-{curr}")


    def get_state_type(self) -> StateType:
        """Classify the point by which 3D estimates exist in the previous
        and current frame.
        """
        prev = self._state_kind(self.p_prev, self.v_prev)
        curr = self._state_kind(self.p_curr, self.v_curr)
        return StateType(f"{prev}-{curr}")


    # ---- internal helpers ----------------------------------------------------

    @staticmethod
    def _slot_kind(uL, uR) -> str:
        """N / M / S based on which stereo slots are populated."""
        has_L = uL is not None
        has_R = uR is not None
        if has_L and has_R: return "S"
        if has_L or  has_R: return "M"
        return "N"

    @staticmethod
    def _state_kind(p, v) -> str:
        """N / P / PV based on which 3D estimates are populated."""
        if p is None: return "N"
        if v is None: return "P"
        return "PV"
    


# =============================================================================
# PointSet — collection with bookkeeping
# =============================================================================

class PointSet:
    """Dict-backed collection of Points keyed by id.

    Three instances live in the pipeline, one per role:
        F       : EKF feature points (inertial references)
        F_pre   : candidates awaiting first solve / admission
        I       : interest points

    Role is implicit in which set a point belongs to.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._points: dict[int, Point] = {}

    # ---- container protocol ----------------------------------------------
    def __len__(self) -> int: ...
    def __contains__(self, pid: int) -> bool: ...
    def __iter__(self) -> Iterator[Point]: ...
    def __getitem__(self, pid: int) -> Point: ...

    # ---- mutation --------------------------------------------------------
    def add(self, p: Point) -> None:
        """Insert a point. Raises if id already present."""
        ...

    def remove(self, pid: int) -> Point:
        """Remove and return a point. Raises if not present."""
        ...

    def discard(self, pid: int) -> None:
        """Remove a point if present, no-op otherwise."""
        ...

    def roll(self) -> None:
        """Call .roll() on every contained point."""
        ...

    # ---- queries ---------------------------------------------------------
    def ids(self) -> set[int]:
        """All currently-held point ids."""
        ...

    def filter(self, predicate) -> "PointSet":
        """Return a new PointSet containing points where predicate(p) is True.
        Does not deep-copy points — both sets share Point references.
        """
        ...
        
    def filter_px_type(self, t: PixelType) -> Iterator[Point]: 
        ...

    def filter_state_type(self, t: StateType) -> Iterator[Point]: 
        ...

    # ---- bulk transfer ---------------------------------------------------
    def move_to(self, other: "PointSet", pids: set[int]) -> None:
        """Move points by id from this set to another (e.g. F_pre → F on
        admission, F → drop on graduation-out, etc.).
        """
        ...


# =============================================================================
# Module-level helpers
# =============================================================================

class IdSource:
    """Monotonic id generator. Single instance shared across the pipeline so
    ids never collide between sets or across re-detections.
    """
    def __init__(self, start: int = 0) -> None: ...
    def next(self) -> int: ...