# ================================================================
#  VEV – Semestrálne zadanie (Por. 31 – Zhumabekov Serikbolsyn)
#  2x400 kV, kotevný úsek – terén z VEV_2025_teren.xlsx
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# 1. VSTUPNÉ DÁTA – TVOJE KONKRÉTNE ZADANIE (Por. 31)
# ---------------------------------------------------------------

def load_input():
    """
    Načíta:
      - špecifikáciu zadania zo specifikacia_zadania_2025.xlsx
      - terén z VEV_2025_teren.xlsx
    a vráti:
      zadanie: dict s parametrami pre ampacitu a zaťaženia
      terrain: DataFrame s profilom terénu
    """

    # Načítaj celý súbor a vyber svoj riadok (Por. = 31)
    spec = pd.read_excel("specifikacia_zadania_2025.xlsx")
    row = spec[spec["Por."] == 31].iloc[0]

    # min vietor je v xls s čiarkou, pre istotu preparse
    def _to_float(v):
        if isinstance(v, str):
            return float(v.replace(",", "."))
        return float(v)

    zadanie = {
        # identifikácia
        "poradie": int(row["Por."]),
        "meno": str(row["Celé meno s titulmi"]),
        "typizacia": str(row["Typizácia"]),            # Mačka
        "uroven_spolahlivosti": int(row["úroveň spolahlivosti"]),  # 2
        "kategoria_terenu": int(row["kategória terenu"]),          # 1
        "vetrova_oblast": int(row["vetrova oblasť"]),              # 2
        "typ_terenu": int(row["typ terenu"]),                      # 2
        "namrazova_oblast": str(row["namrazová oblast"]),          # I3
        "typ_namrazi": str(row["typ námrazi"]),                    # Mokrý sneh

        # okrajové podmienky pre ampacitu
        "T_ambient_max": float(row["max teplota okolia degree C"]),    # 39 °C
        "v_wind_min": _to_float(row["min vietor m/s"]),                # 0.2 m/s
        "G_global": float(row["slečný príkon w/m"]),                   # 1241 W/m2
        "alpha_abs": float(row["koeficient absorbiecie"]),             # 0.5
        "epsilon": float(row["koeficient emisivity"]),                 # 0.5
    }

    # Terénny profil – X[m], Y[m]
    terrain = pd.read_excel("VEV_2025_teren.xlsx")

    # malá kontrola – dĺžka kotevného úseku
    zadanie["L_start"] = float(terrain["X [m]"].min())
    zadanie["L_end"] = float(terrain["X [m]"].max())
    zadanie["L_total"] = zadanie["L_end"] - zadanie["L_start"]

    return zadanie, terrain


# ---------------------------------------------------------------
# 2. VODIČ – TU VYBERIEŠ KONKRÉTNY TYP Z TABULIEK
# ---------------------------------------------------------------

def choose_conductor(zadanie):
    """
    Vyberie vodič tak, aby v trojzväzku preniesol >= 2000 A.
    Mechanické a geometrické parametre doplníš z Bendíka
    (ACSR / AlFe alebo iný typ podľa zoznamu z cvičenia).

    Zatiaľ len šablóna so štruktúrou.
    """

    conductor = {
        "name": "AL1/ACSR-xxx",        # TODO: reálny názov vodiča
        "diameter": None,              # [m]
        "area": None,                  # [m2]
        "RTS": None,                   # [N] alebo [MPa * area]

        # elektrické / tepelné parametre
        "rho": None,                   # rezistivita pri 20 °C
        "alpha_res": None,             # teplotný koeficient odporu

        # radiačné parametre z tvojho zadania:
        "absorptivity": zadanie["alpha_abs"],
        "emissivity": zadanie["epsilon"],
    }

    return conductor


# ---------------------------------------------------------------
# 3. AMPACITA – ŠABLÓNA
# ---------------------------------------------------------------

def compute_ampacity(conductor, zadanie):
    """
    Numerické riešenie Idov z tepelnej rovnice:
      P_Joule(I, T) = P_conv(T, T_ambient, v) + P_rad(T, T_ambient, ε, α, G)

    Vstupy z tvojho zadania:
      T_ambient_max = 39 °C
      v_wind_min    = 0.2 m/s
      G_global      = 1241 W/m2
      α, ε          = 0.5, 0.5

    Zatiaľ len kostra – sem vložíš rovnice z Bendíka.
    """

    T_amb = zadanie["T_ambient_max"]
    v = zadanie["v_wind_min"]
    G = zadanie["G_global"]
    alpha = zadanie["alpha_abs"]
    eps = zadanie["epsilon"]

    # TODO: numerické riešenie pre Idov (napr. bisection / Newton)
    Idov = None

    return Idov


# ---------------------------------------------------------------
# 4. STOŽIARE A ROZPÄTIA – ŠABLÓNA
# ---------------------------------------------------------------

def generate_spans_and_towers(terrain):
    """
    Na základe terénu [X, Y] vytvorí zoznam stožiarov a rozpätí.
    Zatiaľ len úplne jednoduchý placeholder.
    """

    # TODO: nahradiť hrubý odhad reálnym návrhom
    x_start = float(terrain["X [m]"].min())
    x_end = float(terrain["X [m]"].max())

    # napr. tri stožiare: na začiatku, v strede, na konci
    towers = [
        {"id": 1, "x": x_start, "z": float(terrain["Y [m]"].iloc[0]) + 30},
        {"id": 2, "x": 0.5 * (x_start + x_end),
         "z": float(terrain["Y [m]"].iloc[len(terrain)//2]) + 30},
        {"id": 3, "x": x_end, "z": float(terrain["Y [m]"].iloc[-1]) + 30},
    ]

    spans = [
        {"from": towers[0], "to": towers[1]},
        {"from": towers[1], "to": towers[2]},
    ]

    return towers, spans


# ---------------------------------------------------------------
# 5. STAVOVÁ ROVNICA – ŠABLÓNA
# ---------------------------------------------------------------

def solve_state_equation(span, conductor, state):
    """
    Tu implementuješ stavovú rovnicu podľa Bendíka.
    Zatiaľ štruktúra výstupu.
    """

    result = {
        "sigma_h": None,       # MPa
        "c": None,             # m
        "q": None,             # N/m
        "H": None,             # N
        "percent_RTS": None,   # %
        "f": None,             # m – viditeľný priehyb
    }
    return result


# ---------------------------------------------------------------
# 6. MONTÁŽNE TABUĽKY
# ---------------------------------------------------------------

def compute_final_tables(spans, conductor):
    final_states = [-30, -20, -10, -5, "N", "V", "Nv", "nV",
                    0, 10, 20, 30, 40, 60, 80]

    rows = []
    for span in spans:
        for state in final_states:
            r = solve_state_equation(span, conductor, state)
            rows.append({
                "span_id_from": span["from"]["id"],
                "span_id_to": span["to"]["id"],
                "state": state,
                **r
            })
    return pd.DataFrame(rows)


def compute_initial_tables(spans, conductor):
    states = [-10, -5, 0, 10, 15, 17, 20, 22, 25, 27, 30, 35, 40]

    rows = []
    for span in spans:
        for state in states:
            r = solve_state_equation(span, conductor, state)
            rows.append({
                "span_id_from": span["from"]["id"],
                "span_id_to": span["to"]["id"],
                "state": state,
                **r
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 7. VIBRÁCIE – ŠABLÓNA
# ---------------------------------------------------------------

def check_vibrations(span, conductor):
    """
    Jednoduchá kontrola vibrácií – doplníš podľa učebnice.
    """
    return {"OK": True, "krit": None}


# ---------------------------------------------------------------
# 8. MINIMÁLNE PRIBLÍŽENIA – ŠABLÓNA
# ---------------------------------------------------------------

def compute_clearances(spans, terrain):
    """
    Pre každý rozpätie skontroluje min. vzdialenosť vodiča od terénu.
    Zatiaľ len štruktúra.
    """
    result = []

    for span in spans:
        clearance_min = None  # TODO: reálny výpočet
        result.append({
            "span_id_from": span["from"]["id"],
            "span_id_to": span["to"]["id"],
            "min_clearance": clearance_min
        })

    return pd.DataFrame(result)


def plot_profile(spans, terrain):
    """
    Terénny profil + vodič (keď budeš mať c a f).
    """
    plt.figure(figsize=(12, 5))
    plt.plot(terrain["X [m]"], terrain["Y [m]"], label="Terén")

    # TODO: pridať krivku vodiča pre stav max. priehybu

    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title("Priečny profil kotevného úseku")
    plt.grid()
    plt.legend()
    plt.show()


# ---------------------------------------------------------------
# 9. EXPORT
# ---------------------------------------------------------------

def export_results(final_tables, initial_tables, clearances):
    final_tables.to_excel("final_tables.xlsx", index=False)
    initial_tables.to_excel("initial_tables.xlsx", index=False)
    clearances.to_excel("clearances.xlsx", index=False)
    print("Výsledky exportované.")


# ---------------------------------------------------------------
# 10. MAIN
# ---------------------------------------------------------------

def main():
    zadanie, terrain = load_input()
    print("Načítané zadanie:", zadanie)

    conductor = choose_conductor(zadanie)

    print("Výpočet ampacity...")
    Idov = compute_ampacity(conductor, zadanie)
    print("Ampacita Idov =", Idov)

    towers, spans = generate_spans_and_towers(terrain)

    print("Montážne tabuľky...")
    final_tables = compute_final_tables(spans, conductor)
    initial_tables = compute_initial_tables(spans, conductor)

    print("Priblíženia...")
    clearances = compute_clearances(spans, terrain)

    export_results(final_tables, initial_tables, clearances)
    # plot_profile(spans, terrain)  # keď doplníš výpočet vodiča

    print("Hotovo (kostra).")


if __name__ == "__main__":
    main()
