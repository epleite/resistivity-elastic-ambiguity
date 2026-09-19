"""Constitutive model families used by the Phase-0.5 robustness benchmark.

The goal is not to declare one universal rock-physics law.  It is to expose the
identifiability test to several physically distinct, transparent forward maps
and to deliberate truth/inversion mismatch.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .models import Scenario, _gassmann, _vrh

ELASTIC_FAMILIES = ("soft_sand", "stiff_sand", "contact_cement", "dem", "dem_single")
ELECTRICAL_FAMILIES = ("archie", "waxman_smits", "dual_porosity")
COUPLING_MODES = ("rigid", "partial", "independent")


def _mineral_and_fluid(p: Mapping[str, float], scenario: Scenario):
    phi, sw, vcl = p["phi"], p["sw"], p["vcl"]
    kmin = _vrh(vcl, scenario.mineral_k_gpa, scenario.clay_k_gpa)
    gmin = _vrh(vcl, scenario.mineral_g_gpa, scenario.clay_g_gpa)
    rhomin = (1.0 - vcl) * scenario.mineral_rho + vcl * scenario.clay_rho
    kbr, khc = 2.8, 0.12
    rhobr, rhohc = 1.05, 0.72
    kfl = 1.0 / (sw / kbr + (1.0 - sw) / khc)
    rhofl = sw * rhobr + (1.0 - sw) * rhohc
    return phi, kmin, gmin, rhomin, kfl, rhofl


def _hertz_mindlin(kmin: float, gmin: float, coord: float, phi_c: float = 0.40):
    """Hertz-Mindlin endpoint at critical porosity (20 MPa effective stress)."""
    pressure = 0.020  # GPa
    nu = (3.0 * kmin - 2.0 * gmin) / (2.0 * (3.0 * kmin + gmin))
    khm = (
        coord**2 * (1.0 - phi_c) ** 2 * gmin**2 * pressure
        / (18.0 * np.pi**2 * (1.0 - nu) ** 2)
    ) ** (1.0 / 3.0)
    ghm = (5.0 - 4.0 * nu) / (5.0 * (2.0 - nu)) * (
        3.0 * coord**2 * (1.0 - phi_c) ** 2 * gmin**2 * pressure
        / (2.0 * np.pi**2 * (1.0 - nu) ** 2)
    ) ** (1.0 / 3.0)
    return khm, ghm


def _hs_interpolation(
    phi: float, kmin: float, gmin: float, khm: float, ghm: float, mode: str,
    phi_c: float = 0.40,
):
    """Modified lower (soft) or upper (stiff) Hashin-Shtrikman interpolation."""
    t = np.clip(phi / phi_c, 0.0, 0.995)
    if mode == "soft_sand":
        gref = ghm
    elif mode == "stiff_sand":
        gref = gmin
    else:
        raise ValueError(f"Unknown interpolation mode: {mode}")
    zref = gref / 6.0 * (9.0 * (khm if mode == "soft_sand" else kmin) + 8.0 * gref) / (
        (khm if mode == "soft_sand" else kmin) + 2.0 * gref
    )
    kdry = 1.0 / (
        t / (khm + 4.0 * gref / 3.0)
        + (1.0 - t) / (kmin + 4.0 * gref / 3.0)
    ) - 4.0 * gref / 3.0
    gdry = 1.0 / (
        t / (ghm + zref) + (1.0 - t) / (gmin + zref)
    ) - zref
    return max(kdry, 0.02), max(gdry, 0.01)


def _contact_cement_endpoint(
    kmin: float, gmin: float, coord: float, cement: float, phi_c: float = 0.40,
):
    """Dvorkin-style contact-cement endpoint (scheme-1 contact thickness)."""
    # Cement is kept slightly softer than the VRH mineral mixture for numerical
    # robustness across quartz-rich and calcitic scenarios.
    kc, gc = 0.80 * kmin, 0.55 * gmin
    num = max(2.0 * cement, 1e-9)
    den = max(3.0 * coord * (1.0 - phi_c), 1e-9)
    alpha = (num / den) ** 0.25
    nu_m = (3.0 * kmin - 2.0 * gmin) / (2.0 * (3.0 * kmin + gmin))
    nu_c = (3.0 * kc - 2.0 * gc) / (2.0 * (3.0 * kc + gc))
    lambda_n = (
        2.0 * gc * (1.0 - nu_m) * (1.0 - nu_c)
        / (np.pi * gmin * max(1.0 - 2.0 * nu_c, 1e-5))
    )
    an = -0.024153 * lambda_n ** -1.3646
    bn = 0.20405 * lambda_n ** -0.89008
    cn = 0.00024649 * lambda_n ** -1.9864
    sn = an * alpha**2 + bn * alpha + cn

    lambda_t = gc / (np.pi * gmin)
    at = -1e-2 * (2.26 * nu_m**2 + 2.07 * nu_m + 2.30) * lambda_t ** (
        0.079 * nu_m**2 + 0.1754 * nu_m - 1.342
    )
    bt = (0.0573 * nu_m**2 + 0.0937 * nu_m + 0.202) * lambda_t ** (
        0.0274 * nu_m**2 + 0.0529 * nu_m - 0.8765
    )
    ct = 1e-4 * (9.654 * nu_m**2 + 4.945 * nu_m + 3.10) * lambda_t ** (
        0.01867 * nu_m**2 + 0.4011 * nu_m - 1.8186
    )
    st = at * alpha**2 + bt * alpha + ct
    mc = kc + 4.0 * gc / 3.0
    kdry = coord * (1.0 - phi_c) * mc * sn / 6.0
    gdry = 3.0 * kdry / 5.0 + 3.0 * coord * (1.0 - phi_c) * gc * st / 20.0
    return max(kdry, 0.02), max(gdry, 0.01)


def _dem_dry(
    phi: float, secondary: float, aspect: float, kmin: float, gmin: float,
):
    """Differential-effective-medium dry frame with a pore-shape correction.

    The spherical dry-pore DEM equations are integrated explicitly.  A bounded
    shape multiplier makes secondary/compliant pores more damaging; this is a
    transparent DEM-inspired approximation, not the full spheroidal tensor.
    """
    secondary = np.clip(secondary, 0.0, 0.85 * phi)
    primary = max(phi - secondary, 0.0)
    primary_shape = np.clip(1.0 + 0.12 * (0.10 / max(aspect, 0.012) - 1.0), 0.7, 2.5)
    secondary_shape = np.clip(0.10 / max(aspect, 0.012), 0.8, 6.0)
    damage_phi = np.clip(primary * primary_shape + secondary * secondary_shape, 0.0, 0.52)
    k, g = float(kmin), float(gmin)
    steps = 36
    dphi = damage_phi / steps
    occupied = 0.0
    for _ in range(steps):
        bulk_den = max(4.0 * g / 3.0, 1e-8)
        f = g * (9.0 * k + 8.0 * g) / max(6.0 * (k + 2.0 * g), 1e-8)
        dk = -k * (k + 4.0 * g / 3.0) / bulk_den / max(1.0 - occupied, 0.05)
        dg = -g * (g + f) / max(f, 1e-8) / max(1.0 - occupied, 0.05)
        k = max(k + dk * dphi, 0.015)
        g = max(g + dg * dphi, 0.008)
        occupied += dphi
    return k, g


def forward_elastic_family(
    p: Mapping[str, float], scenario: Scenario, family: str,
) -> np.ndarray:
    """Return Vp, Vs and density for one elastic constitutive family."""
    if family not in ELASTIC_FAMILIES:
        raise ValueError(f"Unknown elastic family: {family}")
    phi, kmin, gmin, rhomin, kfl, rhofl = _mineral_and_fluid(p, scenario)
    coord = p["coord"]
    cement = p["cement"]
    aspect = p["aspect"]
    secondary = p.get("secondary", 0.0)
    khm, ghm = _hertz_mindlin(kmin, gmin, coord)

    if family in ("soft_sand", "stiff_sand"):
        kdry, gdry = _hs_interpolation(phi, kmin, gmin, khm, ghm, family)
        kc, gc = _contact_cement_endpoint(kmin, gmin, coord, cement)
        cement_weight = np.clip(cement / 0.12, 0.0, 0.75)
        kdry = (1.0 - cement_weight) * kdry + cement_weight * kc
        gdry = (1.0 - cement_weight) * gdry + cement_weight * gc
        # Small crack-compliance correction; it is intentionally weaker than DEM.
        crack = np.clip(1.0 - 0.18 * phi * (0.10 / max(aspect, 0.012) - 1.0), 0.55, 1.15)
        kdry *= crack
        gdry *= np.sqrt(crack)
    elif family == "contact_cement":
        kc, gc = _contact_cement_endpoint(kmin, gmin, coord, cement)
        # Constant-cement-like porosity interpolation away from the cemented endpoint.
        porosity_factor = np.clip((1.0 - phi / 0.40) / 0.55, 0.12, 1.6)
        kdry, gdry = kc * porosity_factor**0.75, gc * porosity_factor**0.85
    else:  # simultaneous multimodal DEM or deliberately misspecified single-pore DEM
        if family == "dem_single":
            kdry, gdry = _dem_dry(phi, 0.0, 0.15, kmin, gmin)
        else:
            kdry, gdry = _dem_dry(phi, secondary, aspect, kmin, gmin)
        cement_weight = np.clip(cement / 0.15, 0.0, 0.55)
        kc, gc = _contact_cement_endpoint(kmin, gmin, coord, cement)
        kdry = (1.0 - cement_weight) * kdry + cement_weight * kc
        gdry = (1.0 - cement_weight) * gdry + cement_weight * gc

    kdry = float(np.clip(kdry, 0.01, 0.97 * kmin))
    gdry = float(np.clip(gdry, 0.006, 0.97 * gmin))
    ksat = _gassmann(kdry, kmin, kfl, phi)
    rho = (1.0 - phi) * rhomin + phi * rhofl
    vp = np.sqrt((ksat + 4.0 * gdry / 3.0) / rho)
    vs = np.sqrt(gdry / rho)
    return np.array([vp, vs, rho], dtype=float)


def _connected_porosity(
    p: Mapping[str, float], coupling: str,
) -> tuple[float, float, float]:
    """Return primary, connected-secondary, and total connected porosity."""
    if coupling not in COUPLING_MODES:
        raise ValueError(f"Unknown coupling mode: {coupling}")
    phi = p["phi"]
    secondary = np.clip(p.get("secondary", 0.0), 0.0, 0.85 * phi)
    primary = max(phi - secondary, 0.005)
    if coupling == "rigid":
        sec_connected = secondary
        total = primary + sec_connected
    elif coupling == "partial":
        sec_connected = 0.35 * secondary
        total = primary + sec_connected
    else:
        sec_connected = 0.0
        total = np.clip(p.get("phi_e", phi), 0.01, 0.50)
        primary = total
    return primary, sec_connected, np.clip(total, 0.01, 0.50)


def forward_electrical_family(
    p: Mapping[str, float], scenario: Scenario, family: str,
    coupling: str = "partial",
) -> np.ndarray:
    """Return deep and shallow resistivity for an electrical model family."""
    if family not in ELECTRICAL_FAMILIES:
        raise ValueError(f"Unknown electrical family: {family}")
    primary, secondary, phi_conn = _connected_porosity(p, coupling)
    sw = np.clip(p["sw"], 0.01, 0.999)
    rw = np.exp(p["log_rw"])
    m, n = p["archie_m"], p["archie_n"]
    vcl = p["vcl"]
    surface = max(p["surface_cond"], 0.0)
    invasion = np.clip(p.get("invasion", 0.20), 0.0, 0.65)

    def conductivity(saturation: float, shallow: bool) -> float:
        archie = phi_conn**m * saturation**n / rw
        if family == "archie":
            sigma = archie
        elif family == "waxman_smits":
            # Simplified Waxman-Smits-style parallel clay/water conduction.
            clay_term = surface * vcl * (0.35 + saturation) / max(phi_conn, 0.03)
            sigma = archie + clay_term
        else:
            macro = primary**m * saturation**n / rw
            micro_m = np.clip(m - 0.35, 1.2, 3.2)
            micro_n = np.clip(n + 0.20, 1.3, 3.5)
            micro = 0.65 * secondary**micro_m * saturation**micro_n / rw
            clay_term = 0.55 * surface * vcl * (0.35 + saturation) / max(phi_conn, 0.03)
            sigma = macro + micro + clay_term
        if shallow and family != "archie":
            sigma += 0.15 * surface * vcl
        return max(sigma, 1e-10)

    sw_shallow = np.clip(sw + invasion * (1.0 - sw), 0.01, 0.999)
    deep = conductivity(sw, False)
    shallow = conductivity(sw_shallow, True)
    return 1.0 / np.array([deep, shallow], dtype=float)
