# VEV Semester Project - Por. 31 - 2x400 kV overhead line

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import bisect


def load_input():
    """Load assignment and terrain data."""
    spec = pd.read_excel("specifikacia_zadania_2025.xlsx")
    row = spec[spec["Por."] == 31].iloc[0]

    def _to_float(v):
        if isinstance(v, str):
            return float(v.replace(",", "."))
        return float(v)

    zadanie = {
        "poradie": int(row["Por."]),
        "meno": str(row["Celé meno s titulmi"]),
        "typizacia": str(row["Typizácia"]),
        "uroven_spolahlivosti": int(row["úroveň spolahlivosti"]),
        "kategoria_terenu": int(row["kategória terenu"]),
        "vetrova_oblast": int(row["vetrova oblasť"]),
        "typ_terenu": int(row["typ terenu"]),
        "namrazova_oblast": str(row["namrazová oblast"]),
        "typ_namrazi": str(row["typ námrazi"]),

        "T_ambient_max": float(row["max teplota okolia degree C"]),
        "v_wind_min": _to_float(row["min vietor m/s"]),
        "G_global": float(row["slečný príkon w/m"]),
        "alpha_abs": float(row["koeficient absorbiecie"]),
        "epsilon": float(row["koeficient emisivity"]),
    }

    terrain = pd.read_excel("VEV_2025_teren.xlsx")

    zadanie["L_start"] = float(terrain["X [m]"].min())
    zadanie["L_end"] = float(terrain["X [m]"].max())
    zadanie["L_total"] = zadanie["L_end"] - zadanie["L_start"]

    return zadanie, terrain


def choose_conductor(zadanie):
    """Select conductor: 434-AL1/56-ST1A (STN EN 50182)."""

    conductor = {
        "name": "434-AL1/56-ST1A",
        "diameter": 28.80e-3,
        "area_total": 490.59e-6,
        "area_Al": None,
        "mass_per_m": 1.6413,
        "RTS": 133.59e3,
        "E": 70_491e6,
        "R20": 0.0666 / 1000,
        "Rac20": 0.0710 / 1000,
        "Rac80": 0.0875 / 1000,
        "alpha_res": 19.44e-6,
        "alpha_lin": 19.3e-6,
        "absorptivity": zadanie["alpha_abs"],
        "emissivity": zadanie["epsilon"],
    }

    print(f"\nConductor: {conductor['name']}")
    print(f"Diameter = {conductor['diameter']} m")
    print(f"RTS = {conductor['RTS']/1000} kN\n")

    return conductor


def air_properties(T_film):
    """Temperature-dependent air properties."""
    T_K = T_film + 273.15
    mu = (1.458e-6 * T_K**1.5) / (T_K + 110.4)
    k = 0.024 + 7.0e-5 * T_film
    rho = 101325.0 / (287.05 * T_K)
    nu = mu / rho
    alpha_air = k / (rho * 1005.0)
    Pr = nu / alpha_air
    return mu, k, rho, nu, Pr


def resistance_at_temp(R20, alpha_res, T):
    """Resistance at temperature T."""
    return R20 * (1.0 + alpha_res * (T - 20.0))


def q_solar(D, G, alpha_abs, theta=90.0):
    """Solar heat gain [W/m]: Qs = α*G*D*sin(θ). Use θ=90° for max heating."""
    return alpha_abs * G * D * np.sin(np.deg2rad(theta))


def q_radiation(D, T_cond, T_amb, epsilon):
    """Radiative heat loss [W/m]."""
    sigma_sb = 5.670374419e-8
    A_per_m = np.pi * D
    T_c_K = T_cond + 273.15
    T_a_K = T_amb + 273.15
    return epsilon * sigma_sb * A_per_m * (T_c_K**4 - T_a_K**4)


def q_convection(D, T_cond, T_amb, v_wind):
    """Convective heat loss [W/m] using Churchill-Bernstein."""
    if T_cond <= T_amb:
        return 0.0

    T_film = 0.5 * (T_cond + T_amb)
    mu, k_air, rho_air, nu_air, Pr = air_properties(T_film)

    if v_wind < 1e-6:
        v_wind = 0.01

    Re = (v_wind * D) / nu_air
    if Re < 1e-3:
        Re = 1e-3

    Nu = 0.3 + (
        (0.62 * Re**0.5 * Pr**(1.0 / 3.0))
        / (1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25
    ) * (1.0 + (Re / 282000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0)

    h = Nu * k_air / D
    A_per_m = np.pi * D
    return h * A_per_m * (T_cond - T_amb)


def compute_ampacity(conductor, zadanie):
    T_cond = 80.0
    T_amb = zadanie["T_ambient_max"]
    v = zadanie["v_wind_min"]
    G = zadanie["G_global"]
    alpha = zadanie["alpha_abs"]
    eps = zadanie["epsilon"]

    D = conductor["diameter"]
    R20 = conductor["R20"]
    alpha_res = conductor["alpha_res"]

    R_T = resistance_at_temp(R20, alpha_res, T_cond)
    Qs = q_solar(D, G, alpha, theta=90.0)
    Qr = lambda: q_radiation(D, T_cond, T_amb, eps)
    Qc = lambda: q_convection(D, T_cond, T_amb, v)

    def heat_balance(I):
        P_j = (I ** 2) * R_T
        return P_j + Qs - (Qc() + Qr())

    I_min, I_max = 0.0, 4000.0
    f_min, f_max = heat_balance(I_min), heat_balance(I_max)
    if f_min * f_max > 0:
        print("Pozor: heat_balance(I_min) a heat_balance(I_max) majú rovnaké znamenko.")
        return None

    for _ in range(60):
        I_mid = 0.5 * (I_min + I_max)
        f_mid = heat_balance(I_mid)
        if f_min * f_mid <= 0:
            I_max, f_max = I_mid, f_mid
        else:
            I_min, f_min = I_mid, f_mid
        if abs(I_max - I_min) < 1e-3:
            break

    return 0.5 * (I_min + I_max)


def interp_terrain_z(x, terrain):
    """Interpolate terrain elevation at position x."""
    return float(np.interp(x, terrain["X [m]"], terrain["Y [m]"]))


def generate_spans_and_towers(terrain, base_head_height=30.0):
    """Generate 3 towers and 2 spans for anchor section."""
    x_start = float(terrain["X [m]"].min())
    x_end = float(terrain["X [m]"].max())
    x_mid = 0.5 * (x_start + x_end)
    z_start_terr = interp_terrain_z(x_start, terrain)
    z_mid_terr = interp_terrain_z(x_mid, terrain)
    z_end_terr = interp_terrain_z(x_end, terrain)

    towers = [
        {"id": 1, "x": x_start, "z": z_start_terr + base_head_height},
        {"id": 2, "x": x_mid,   "z": z_mid_terr   + base_head_height},
        {"id": 3, "x": x_end,   "z": z_end_terr   + base_head_height},
    ]

    spans = [
        {"from": towers[0], "to": towers[1]},
        {"from": towers[1], "to": towers[2]},
    ]

    return towers, spans


def compute_ice_load(conductor, zadanie):
    """Ice load q_ice [N/m] per STN EN 50341-3-22. Wet snow: 400 kg/m³."""
    g = 9.81
    D = conductor["diameter"]
    ice_thickness_map = {
        "I1": 0.0,
        "I2": 0.012,
        "I3": 0.020,
        "I4": 0.028,
        "I5": 0.036,
    }

    region = zadanie["namrazova_oblast"]
    t_ice = ice_thickness_map.get(region, 0.0)
    if t_ice <= 0.0:
        return 0.0

    typ_namrazi = zadanie.get("typ_namrazi", "")
    if "Mokrý" in typ_namrazi or "sneh" in typ_namrazi:
        rho_ice = 400.0
    else:
        rho_ice = 900.0

    D_i = D + 2.0 * t_ice
    A_ice = 0.25 * np.pi * (D_i**2 - D**2)
    q_ice = A_ice * rho_ice * g
    return q_ice


def compute_wind_load(conductor, zadanie, with_ice=False):
    """Wind load q_wind [N/m] per STN EN 50341: q = qb*ce*Cx*cscd*D."""
    D = conductor["diameter"]
    qb_ref_map = {
        1: 360.0,
        2: 450.0,
        3: 560.0,
        4: 680.0,
    }
    qb_ref = qb_ref_map.get(zadanie["vetrova_oblast"], 450.0)

    kategoria = zadanie.get("kategoria_terenu", 1)
    ce_map = {
        1: 2.4,
        2: 2.0,
        3: 1.6,
        4: 1.3,
    }
    ce = ce_map.get(kategoria, 2.4)

    if with_ice:
        Cx = 1.2
        ice_thickness_map = {"I1": 0.0, "I2": 0.012, "I3": 0.020, "I4": 0.028, "I5": 0.036}
        t_ice = ice_thickness_map.get(zadanie["namrazova_oblast"], 0.0)
        D_effective = D + 2.0 * t_ice
    else:
        Cx = 1.0
        D_effective = D

    cscd = 0.95
    p_wind = qb_ref * ce * Cx * cscd
    q_wind = p_wind * D_effective
    return q_wind


def compute_load_per_meter(conductor, zadanie, state):
    """Returns (q_vert, q_horiz, q_eq) in N/m."""
    g = 9.81
    q_g = conductor["mass_per_m"] * g
    q_ice = 0.0
    q_wind = 0.0
    has_ice = False

    if isinstance(state, (int, float)):
        pass
    else:
        s = str(state).upper()
        if "N" in s:
            has_ice = True
            q_ice = compute_ice_load(conductor, zadanie)
        if "V" in s:
            q_wind = compute_wind_load(conductor, zadanie, with_ice=has_ice)

    q_vert = q_g + q_ice
    q_horiz = q_wind
    q_eq = (q_vert**2 + q_horiz**2) ** 0.5
    return q_vert, q_horiz, q_eq


def solve_state_equation(span, conductor, zadanie, state):
    """Solve state equation for horizontal stress."""
    T_ref = 15.0
    sigma_ref_MPa = 81.67
    sigma_ref = sigma_ref_MPa * 1e6

    x_from = span["from"]["x"]
    x_to = span["to"]["x"]
    L = abs(x_to - x_from)

    A = conductor["area_total"]
    E = conductor["E"]
    alpha = conductor["alpha_lin"]

    if isinstance(state, (int, float)):
        T = float(state)
    else:
        T = -5.0

    q_ref_vert, _, _ = compute_load_per_meter(conductor, zadanie, T_ref)
    q_vert, q_horiz, q_eq = compute_load_per_meter(conductor, zadanie, state)

    def state_equation_residual(sigma_h):
        """Length change equation residual."""
        delta_T = T - T_ref
        dL_thermal = L * alpha * delta_T
        dL_stress = L * (sigma_h - sigma_ref) / E
        if sigma_h > 0:
            dL_sag = L * (q_vert**2 * L**2) / (24.0 * sigma_h**2 * E)
            dL_sag_ref = L * (q_ref_vert**2 * L**2) / (24.0 * sigma_ref**2 * E)
        else:
            dL_sag = dL_sag_ref = 0.0
        return dL_thermal + dL_stress + (dL_sag - dL_sag_ref)

    sigma_min = 1e6
    sigma_max = 200e6
    try:
        sigma_h = bisect(state_equation_residual, sigma_min, sigma_max, xtol=1e3)
    except:
        sigma_h = sigma_ref

    sigma_h_MPa = sigma_h / 1e6
    H = sigma_h * A
    c = H / q_vert if q_vert > 0 else 1e6

    if L / c < 0.5:
        f = q_vert * L**2 / (8.0 * H)
    else:
        f = c * (np.cosh(L / (2.0 * c)) - 1.0)

    percent_RTS = 100.0 * H / conductor["RTS"]

    return {
        "sigma_h": sigma_h_MPa, "c": c, "q": q_vert, "q_eq": q_eq,
        "H": H, "percent_RTS": percent_RTS, "f": f, "T": T
    }


def compute_final_tables(spans, conductor, zadanie):
    final_states = [-30, -20, -10, -5, "N", "V", "Nv", "nV", 0, 10, 20, 30, 40, 60, 80]
    rows = []
    for span in spans:
        for state in final_states:
            r = solve_state_equation(span, conductor, zadanie, state)
            rows.append({
                "span_id_from": span["from"]["id"],
                "span_id_to": span["to"]["id"],
                "state": state,
                **r
            })
    return pd.DataFrame(rows)


def compute_initial_tables(spans, conductor, zadanie):
    states = [-10, -5, 0, 10, 15, 17, 20, 22, 25, 27, 30, 35, 40]
    rows = []
    for span in spans:
        for state in states:
            r = solve_state_equation(span, conductor, zadanie, state)
            rows.append({
                "span_id_from": span["from"]["id"],
                "span_id_to": span["to"]["id"],
                "state": state,
                **r
            })
    return pd.DataFrame(rows)


def compute_clearances(spans, terrain):
    """Min clearance calculation."""
    result = []
    for span in spans:
        clearance_min = None
        result.append({
            "span_id_from": span["from"]["id"],
            "span_id_to": span["to"]["id"],
            "min_clearance": clearance_min
        })
    return pd.DataFrame(result)


def export_results(final_tables, initial_tables, clearances):
    final_tables.to_excel("final_tables.xlsx", index=False)
    initial_tables.to_excel("initial_tables.xlsx", index=False)
    clearances.to_excel("clearances.xlsx", index=False)
    print("Results exported.")


def main():
    zadanie, terrain = load_input()
    print("Načítané zadanie:", zadanie)

    conductor = choose_conductor(zadanie)

    print("Výpočet ampacity...")
    Idov = compute_ampacity(conductor, zadanie)
    print("Ampacita Idov =", Idov)

    towers, spans = generate_spans_and_towers(terrain)

    print("Montážne tabuľky...")
    final_tables = compute_final_tables(spans, conductor, zadanie)
    initial_tables = compute_initial_tables(spans, conductor, zadanie)

    print("Priblíženia...")
    clearances = compute_clearances(spans, terrain)

    export_results(final_tables, initial_tables, clearances)

    print("Hotovo (kostra).")


if __name__ == "__main__":
    main()
