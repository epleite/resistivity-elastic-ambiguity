"""Minimal differentiable rock-physics models for the Phase-0 benchmark.

These equations are intentionally transparent rather than field-calibrated.
They are suitable for identifiability experiments, not reservoir estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    mineral_k_gpa: float
    mineral_g_gpa: float
    mineral_rho: float
    clay_k_gpa: float
    clay_g_gpa: float
    clay_rho: float
    truth: Mapping[str, float]


COMMON = {
    "phi": 0.22,
    "sw": 0.55,
    "vcl": 0.05,
    "cement": 0.035,
    "coord": 8.0,
    "aspect": 0.12,
    "secondary": 0.03,
    "log_rw": np.log(0.08),
    "archie_m": 1.9,
    "archie_n": 2.0,
    "surface_cond": 0.010,
}

SCENARIOS = {
    "clean_sandstone": Scenario(
        "clean_sandstone", "Clean sandstone", 37.0, 44.0, 2.65,
        20.9, 6.9, 2.58, {**COMMON, "vcl": 0.02, "surface_cond": 0.001},
    ),
    "clayey_sandstone": Scenario(
        "clayey_sandstone", "Clay-bearing sandstone", 37.0, 44.0, 2.65,
        20.9, 6.9, 2.58, {**COMMON, "vcl": 0.22, "surface_cond": 0.030},
    ),
    "multimodal_carbonate": Scenario(
        "multimodal_carbonate", "Multimodal carbonate", 76.8, 32.0, 2.71,
        37.0, 15.0, 2.70,
        {**COMMON, "phi": 0.18, "vcl": 0.03, "cement": 0.06,
         "coord": 7.0, "aspect": 0.055, "secondary": 0.07,
         "archie_m": 2.15, "surface_cond": 0.002},
    ),
}


def _vrh(v: float, a: float, b: float) -> float:
    """Two-component Voigt-Reuss-Hill average."""
    voigt = (1.0 - v) * a + v * b
    reuss = 1.0 / ((1.0 - v) / a + v / b)
    return 0.5 * (voigt + reuss)


def _gassmann(kdry: float, kmin: float, kfl: float, phi: float) -> float:
    num = (1.0 - kdry / kmin) ** 2
    den = phi / kfl + (1.0 - phi) / kmin - kdry / (kmin * kmin)
    return kdry + num / max(den, 1e-10)


def forward_elastic(p: Mapping[str, float], scenario: Scenario) -> np.ndarray:
    """Return Vp (km/s), Vs (km/s), and bulk density (g/cc).

    The dry frame has explicit coordination, cement and compliant-pore terms.
    Secondary porosity is especially compliant in the carbonate scenario.
    """
    phi, sw, vcl = p["phi"], p["sw"], p["vcl"]
    kmin = _vrh(vcl, scenario.mineral_k_gpa, scenario.clay_k_gpa)
    gmin = _vrh(vcl, scenario.mineral_g_gpa, scenario.clay_g_gpa)
    rhomin = (1.0 - vcl) * scenario.mineral_rho + vcl * scenario.clay_rho

    # Brine/hydrocarbon mixture: Wood average and linear density mixing.
    kbr, khc = 2.8, 0.12
    rhobr, rhohc = 1.05, 0.72
    kfl = 1.0 / (sw / kbr + (1.0 - sw) / khc)
    rhofl = sw * rhobr + (1.0 - sw) * rhohc

    phi_c = 0.40
    contact = np.clip((p["coord"] / 9.0) ** 0.55, 0.35, 1.5)
    cement = 1.0 + 5.0 * np.sqrt(max(p["cement"], 1e-8))
    compliant = 1.0 / (1.0 + 1.7 * phi / max(p["aspect"], 0.012))
    secondary = p.get("secondary", 0.0)
    multi = np.exp(-1.2 * secondary / max(p["aspect"], 0.012))
    frame = np.clip((1.0 - phi / phi_c) ** 2.1 * contact * cement * compliant * multi,
                    0.002, 0.92)
    kdry = kmin * frame
    gdry = gmin * np.clip(frame ** 0.82, 0.005, 0.95)
    ksat = _gassmann(kdry, kmin, kfl, phi)
    rho = (1.0 - phi) * rhomin + phi * rhofl
    vp = np.sqrt((ksat + 4.0 * gdry / 3.0) / rho)
    vs = np.sqrt(gdry / rho)
    return np.array([vp, vs, rho], dtype=float)


def forward_electrical(p: Mapping[str, float], scenario: Scenario) -> np.ndarray:
    """Return deep and shallow resistivity (ohm m).

    Deep conductivity combines an Archie-like connected-water term and surface
    conduction. Shallow Rt uses a deterministic invaded-zone saturation offset;
    this second curve is included to test whether an additional electrical
    observation changes identifiability.
    """
    phi, sw, vcl = p["phi"], p["sw"], p["vcl"]
    rw = np.exp(p["log_rw"])
    m, n = p["archie_m"], p["archie_n"]
    secondary = p.get("secondary", 0.0)
    # Secondary pores contribute only partly to the connected electrical network.
    phi_conn = np.clip(phi - 0.65 * secondary, 0.015, 0.55)
    sigma_surface = p["surface_cond"] * vcl * (1.0 - sw + 0.25)
    sigma_deep = phi_conn**m * sw**n / rw + sigma_surface
    sw_shallow = np.clip(sw + 0.20 * (1.0 - sw), 0.01, 0.995)
    sigma_shallow = phi_conn**m * sw_shallow**n / rw + 1.25 * sigma_surface
    return 1.0 / np.array([sigma_deep, sigma_shallow], dtype=float)
