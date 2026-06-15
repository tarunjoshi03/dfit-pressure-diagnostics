"""
Fracture design parameters derived from DFIT interpretation.

This module closes the loop between a DFIT and the follow-up hydraulic
fracture treatment. The DFIT gives you the rock properties; this module
translates those properties into the treatment design parameters that go
into a fracture simulator.

The workflow:

    DFIT outputs (ISIP, closure pressure, net pressure, closure time)
           |
           v
    Leakoff coefficient  C_L  (fluid loss rate per unit area)
    Fluid efficiency     eta  (fraction of injected fluid that stays in fracture)
    Fracture half-length x_f  (PKN geometry estimate)
    Treatment pressure   p_treat (minimum surface treating pressure)
    Fracture width       w_avg   (average width at design net pressure)
           |
           v
    Design recommendations: rate, volume, pressure, proppant schedule

All equations follow the PKN (Perkins-Kern-Nordgren) fracture geometry model,
which is the standard for design in moderate-to-high-net-pressure systems.
Carter's leakoff model is used for the leakoff coefficient.

References
----------
Nolte, K.G. (1979) SPE-8341-MS.
Economides, M.J. and Nolte, K.G. (2000) "Reservoir Stimulation", 3rd ed.
    Wiley. Chapters 5-9.
Barree, R.D., Miskimins, J.L., Gilbert, J.V. (2015) SPE-169539-PA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FractureDesignParams:
    """
    Treatment-design parameters derived from a DFIT interpretation.

    All pressure values are in psi, lengths in feet, time in minutes,
    volumes in barrels, and rates in bbl/min.
    """
    # --- leakoff / efficiency ---
    leakoff_coefficient_ft_sqrtmin: float
    fluid_efficiency: float
    fracture_closure_time_min: float

    # --- geometry (PKN) ---
    fracture_half_length_ft: float
    average_width_in: float
    fracture_height_ft: float

    # --- treatment pressures ---
    minimum_treating_pressure_psi: float
    bhtp_at_design_rate_psi: float
    surface_treating_pressure_psi: float

    # --- volume / rate ---
    design_rate_bpm: float
    total_fluid_volume_bbl: float
    pad_volume_bbl: float

    # --- inputs carried through for reference ---
    isip_psi: float
    closure_pressure_psi: float
    net_pressure_psi: float
    tvd_ft: float

    note: str = ""


def fracture_design_from_dfit(
    isip_psi: float,
    closure_pressure_psi: float,
    closure_time_min: float,
    t_pump_min: float,
    fracture_height_ft: float = 100.0,
    youngs_modulus_psi: float = 4e6,
    design_half_length_ft: float = 500.0,
    design_rate_bpm: float = 10.0,
    fluid_viscosity_cp: float = 1.0,
    tvd_ft: float = 8000.0,
    surface_pipe_friction_psi: float = 500.0,
    perforation_friction_psi: float = 200.0,
) -> FractureDesignParams:
    """
    Derive hydraulic fracture treatment design parameters from DFIT results.

    Parameters
    ----------
    isip_psi : float
        Instantaneous shut-in pressure from DFIT (psi).
    closure_pressure_psi : float
        Fracture closure pressure from DFIT = minimum in-situ stress (psi).
    closure_time_min : float
        Time from shut-in to fracture closure (minutes).
    t_pump_min : float
        DFIT pumping time (minutes).
    fracture_height_ft : float
        Assumed or known fracture height (ft). Default 100 ft.
    youngs_modulus_psi : float
        Plane-strain Young's modulus of the formation (psi). Default 4e6 psi.
    design_half_length_ft : float
        Target fracture half-length for the main treatment (ft).
    design_rate_bpm : float
        Planned injection rate for the main treatment (bbl/min).
    fluid_viscosity_cp : float
        Treatment fluid viscosity (cp). Default 1.0 cp (slickwater).
    tvd_ft : float
        True vertical depth to the perforations (ft).
    surface_pipe_friction_psi : float
        Estimated surface pipe friction at design rate (psi).
    perforation_friction_psi : float
        Estimated perforation friction at design rate (psi).

    Returns
    -------
    FractureDesignParams
    """
    net_pressure = isip_psi - closure_pressure_psi

    # -----------------------------------------------------------------------
    # 1. Fluid efficiency and leakoff coefficient
    # -----------------------------------------------------------------------
    # From Nolte's material balance: at closure, the ratio of closure time to
    # pumping time determines fluid efficiency.
    # eta = 1 - 2*(tc/tp) / (1 + 2*(tc/tp))  [Carter leakoff, constant area]
    r = closure_time_min / t_pump_min
    fluid_efficiency = max(0.05, min(0.99, 1.0 - 2.0 * r / (1.0 + 2.0 * r)))

    # Carter leakoff coefficient C_L from the material balance:
    # V_injected * eta = fracture volume at shut-in
    # V_lost = V_injected * (1 - eta) = 2 * C_L * A_frac * sqrt(t_pump)
    # For a unit-area fracture: C_L = (1-eta) / (2*sqrt(t_pump)) * [vol/area factor]
    # In field units (ft, min, bbl): C_L [ft/sqrt(min)]
    # We use the dimensionless form and back out C_L from efficiency:
    C_L = (1.0 - fluid_efficiency) / (2.0 * np.sqrt(t_pump_min))

    # -----------------------------------------------------------------------
    # 2. PKN fracture geometry at the design half-length
    # -----------------------------------------------------------------------
    # PKN average fracture width [in]:
    #   w_avg = 0.3 * (q * mu * x_f / (E' * h))^(1/4) * 12   [in field units]
    # where E' = E / (1 - nu^2) ~ E * 1.1 for typical nu = 0.2-0.3
    E_prime = youngs_modulus_psi * 1.1   # plane-strain modulus

    # Unit conversion: q [bbl/min] -> [ft^3/min] = q * 5.615
    q_ft3_min = design_rate_bpm * 5.615
    mu_ft = fluid_viscosity_cp * 2.09e-5  # cp -> lbf*min/ft^2

    w_avg_ft = 0.3 * (
        q_ft3_min * mu_ft * design_half_length_ft / (E_prime * fracture_height_ft)
    ) ** 0.25
    w_avg_in = w_avg_ft * 12.0

    # -----------------------------------------------------------------------
    # 3. Net pressure at design conditions (PKN)
    # -----------------------------------------------------------------------
    # P_net = E' * w_avg / (2 * pi * h / 4)  [simplified PKN]
    design_net_pressure = E_prime * w_avg_ft / (np.pi * fracture_height_ft / 2.0)

    # -----------------------------------------------------------------------
    # 4. Treatment pressures
    # -----------------------------------------------------------------------
    hydrostatic_psi = 0.433 * tvd_ft       # fresh water gradient psi/ft * TVD
    min_treating_pressure = closure_pressure_psi + design_net_pressure
    bhtp = closure_pressure_psi + design_net_pressure
    surface_tp = (bhtp - hydrostatic_psi
                  + surface_pipe_friction_psi + perforation_friction_psi)

    # -----------------------------------------------------------------------
    # 5. Volume and pad design
    # -----------------------------------------------------------------------
    # Total fluid volume from material balance for target half-length:
    # V_total = (2 * h * x_f * w_avg_ft) / (5.615 * eta)  [bbl]
    frac_volume_ft3 = 2.0 * fracture_height_ft * design_half_length_ft * w_avg_ft
    total_volume_bbl = frac_volume_ft3 / (5.615 * fluid_efficiency)

    # Pad volume = fraction needed to create the fracture before adding proppant
    # Rule of thumb: pad fraction ~ (1 - eta)
    pad_volume_bbl = total_volume_bbl * (1.0 - fluid_efficiency)

    note = (
        f"PKN geometry at x_f={design_half_length_ft:.0f} ft, "
        f"h={fracture_height_ft:.0f} ft. "
        f"Fluid efficiency {fluid_efficiency*100:.0f}% from DFIT closure time. "
        f"C_L={C_L:.4f} ft/sqrt(min). "
        f"Net pressure at design conditions: {design_net_pressure:.0f} psi "
        f"({'ABOVE' if design_net_pressure > net_pressure else 'BELOW'} "
        f"DFIT net pressure of {net_pressure:.0f} psi - "
        f"{'adjust rate or height' if design_net_pressure > net_pressure else 'consistent'})."
    )

    return FractureDesignParams(
        leakoff_coefficient_ft_sqrtmin=C_L,
        fluid_efficiency=fluid_efficiency,
        fracture_closure_time_min=closure_time_min,
        fracture_half_length_ft=design_half_length_ft,
        average_width_in=w_avg_in,
        fracture_height_ft=fracture_height_ft,
        minimum_treating_pressure_psi=float(min_treating_pressure),
        bhtp_at_design_rate_psi=float(bhtp),
        surface_treating_pressure_psi=float(surface_tp),
        design_rate_bpm=design_rate_bpm,
        total_fluid_volume_bbl=float(total_volume_bbl),
        pad_volume_bbl=float(pad_volume_bbl),
        isip_psi=isip_psi,
        closure_pressure_psi=closure_pressure_psi,
        net_pressure_psi=float(net_pressure),
        tvd_ft=tvd_ft,
        note=note,
    )


def design_summary(p: FractureDesignParams) -> str:
    """Return a formatted one-page design summary."""
    eta_pct = p.fluid_efficiency * 100
    lines = [
        "FRACTURE DESIGN PARAMETERS  (from DFIT)",
        "=" * 44,
        "",
        "[ DFIT inputs ]",
        f"  ISIP                       {p.isip_psi:>10,.0f}  psi",
        f"  Closure pressure           {p.closure_pressure_psi:>10,.0f}  psi",
        f"  Net pressure (DFIT)        {p.net_pressure_psi:>10,.0f}  psi",
        f"  TVD                        {p.tvd_ft:>10,.0f}  ft",
        "",
        "[ Leakoff & efficiency ]",
        f"  Carter C_L                 {p.leakoff_coefficient_ft_sqrtmin:>10.4f}  ft/sqrt(min)",
        f"  Fluid efficiency           {eta_pct:>10.1f}  %",
        f"  Closure time               {p.fracture_closure_time_min:>10.1f}  min",
        "",
        "[ PKN fracture geometry ]",
        f"  Design half-length         {p.fracture_half_length_ft:>10,.0f}  ft",
        f"  Fracture height            {p.fracture_height_ft:>10,.0f}  ft",
        f"  Average width              {p.average_width_in:>10.3f}  in",
        "",
        "[ Treatment pressures ]",
        f"  Min. treating pressure     {p.minimum_treating_pressure_psi:>10,.0f}  psi (BH)",
        f"  BHTP at design rate        {p.bhtp_at_design_rate_psi:>10,.0f}  psi",
        f"  Surface treating pressure  {p.surface_treating_pressure_psi:>10,.0f}  psi",
        "",
        "[ Volume & rate ]",
        f"  Design rate                {p.design_rate_bpm:>10.1f}  bbl/min",
        f"  Total fluid volume         {p.total_fluid_volume_bbl:>10,.0f}  bbl",
        f"  Pad volume                 {p.pad_volume_bbl:>10,.0f}  bbl",
        "",
        "Note: " + p.note,
    ]
    return "\n".join(lines)
