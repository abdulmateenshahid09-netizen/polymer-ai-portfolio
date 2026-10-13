"""Simulated PE pipe extrusion data. This is DEMO data, not real plant data."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from db import get_connection, init_db, DB_PATH

PRODUCT = "PE100 SDR11 110mm"
LINES = ["Line 1", "Line 2"]
D = 0.110           # outside diameter, m
RHO = 950.0         # density, kg/m3
K = 2.3             # kg/h of output per screw rpm
AIM = 10.45         # wall thickness the operators aim for, mm
LSL, USL = 10.0, 11.1
STEP_MIN = 30       # one reading every 30 minutes
IDEAL_RATE = 1.40   # m/min, best demonstrated line speed (for OEE performance)

OPERATORS = {"Day": "Operator A", "Evening": "Operator B", "Night": "Operator C"}

DOWNTIME_TYPES = [
    # category, weight, (min minutes, max minutes), possible reasons
    ("Changeover", 0.15, (30, 60), ["Size change", "Colour stripe change"]),
    ("Material shortage", 0.15, (10, 40), ["Waiting for resin", "Dryer or loader fault"]),
    ("Mechanical breakdown", 0.15, (20, 120), ["Gearbox fault", "Haul-off fault", "Cutter jam"]),
    ("Die cleaning", 0.20, (15, 30), ["Die lip build-up", "Black specks on pipe"]),
    ("Quality hold", 0.15, (10, 30), ["Wall thickness check", "Waiting for lab result"]),
    ("Minor stoppage", 0.20, (2, 10), ["Pipe jam at cutter", "Sensor reset"]),
]

DEFECT_TYPES = [
    # defect type, expected events per shift
    ("Surface scratches", 0.4),
    ("Black specks", 0.3),
    ("Ovality", 0.25),
    ("Marking error", 0.2),
]

FAULTS = [
    # line, kind, start day, duration hours, description
    ("Line 1", "melt_temp_high", 12.0, 36, "Die heater zone fault, melt temperature +9 C"),
    ("Line 2", "feed_instability", 19.0, 24, "Hopper bridging, unstable output"),
    ("Line 1", "haul_off_slip", 24.0, 16, "Haul-off slipping, speed +5%"),
]


def wall_thickness(output_kgh, haul_mpm):
    """Wall thickness (mm) from a mass balance on the extruded annulus."""
    m = output_kgh / 3600.0
    v = haul_mpm / 60.0
    disc = D**2 - 4 * m / (np.pi * RHO * v)
    return (D - np.sqrt(np.maximum(disc, 0))) / 2 * 1000


def haul_for(output_kgh, thickness_mm):
    """Haul-off speed (m/min) that gives the wanted wall thickness."""
    t = thickness_mm / 1000.0
    m = output_kgh / 3600.0
    v = m / (RHO * np.pi * t * (D - t))
    return v * 60.0


def shift_of(ts):
    h = ts.hour
    if 6 <= h < 14:
        return "Day"
    if 14 <= h < 22:
        return "Evening"
    return "Night"


def fault_at(line, ts, start):
    """Return the fault active on this line at this time, or None."""
    for f_line, kind, start_day, hours, _ in FAULTS:
        if f_line != line:
            continue
        f_start = start + timedelta(days=start_day)
        if f_start <= ts < f_start + timedelta(hours=hours):
            return kind
    return None


def simulate_steps(line, line_no, start, days, rng):
    """Readings every STEP_MIN minutes: measurements, process log, lots."""
    n_steps = int(days * 24 * 60 / STEP_MIN)
    base_out = K * 100
    haul_set = haul_for(base_out, AIM)

    lot_no, lot_left, mfr = 0, 0, 0.30
    lots = {}
    drift, melt_ar = 0.0, 0.0
    recent = []
    meas, proc = [], []
    lot_id = ""

    for i in range(n_steps):
        ts = start + timedelta(minutes=STEP_MIN * i)
        shift = shift_of(ts)
        night = 1.4 if shift == "Night" else 1.0

        if lot_left <= 0:
            lot_no += 1
            lot_id = f"L{line_no}-{lot_no:03d}"
            mfr = float(np.clip(rng.normal(0.30, 0.04), 0.20, 0.40))
            if line == "Line 2" and lot_no == 3:
                mfr = 0.43  # one high-MFR resin lot
            lots[lot_id] = {"mfr": mfr, "start": ts}
            lot_left = int(rng.uniform(72, 120) * 60 / STEP_MIN)
        lot_left -= 1

        fault = fault_at(line, ts, start)
        drift = 0.98 * drift + rng.normal(0, 0.0032 * night)
        melt_ar = 0.9 * melt_ar + rng.normal(0, 0.35)
        melt = 210 + melt_ar + (9.0 if fault == "melt_temp_high" else 0.0)
        rpm = 100 + rng.normal(0, 0.3)
        short_sd = (0.03 if fault == "feed_instability" else 0.004) * night
        eff = 1 + drift + rng.normal(0, short_sd)
        haul_actual = haul_set * (1.05 if fault == "haul_off_slip" else 1.0)

        output = K * rpm * (1 + 0.004 * (melt - 210)) * (1 + 0.15 * (mfr - 0.30)) * eff
        true_t = wall_thickness(output, haul_actual)
        wall = true_t + rng.normal(0, 0.03 * night)
        od = 110.15 + 0.035 * rng.normal() * night

        meas.append((ts, shift, lot_id, "wall_thickness", wall, "mm"))
        meas.append((ts, shift, lot_id, "outside_diameter", od, "mm"))
        proc.append((ts, rpm, haul_actual, melt, mfr, lot_id, shift))

        recent.append(wall)
        every = 8 if shift == "Night" else 4
        if i % every == every - 1:
            dev = float(np.mean(recent[-2:])) - AIM
            if abs(dev) > 0.08:  # operators only react to clear deviations
                haul_set *= 1 + 0.6 * dev / AIM

    return meas, proc, lots


def build_tables(days=30, seed=42):
    rng = np.random.default_rng(seed)
    today = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
    start = today - timedelta(days=days)

    m_rows, p_rows, dt_rows, prod_rows, def_rows, ev_rows = [], [], [], [], [], []

    for line_no, line in enumerate(LINES, start=1):
        meas, proc, lots = simulate_steps(line, line_no, start, days, rng)
        lot_factor = {k: float(rng.lognormal(0, 0.4)) for k in lots}
        bad_lot = list(lots)[2] if line == "Line 2" else None
        if bad_lot:
            lot_factor[bad_lot] *= 4  # contaminated lot

        for ts, shift, lot, ch, val, unit in meas:
            m_rows.append({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "line": line,
                           "product": PRODUCT, "characteristic": ch, "value": round(val, 3),
                           "unit": unit, "shift": shift, "operator": OPERATORS[shift],
                           "lot": lot, "notes": None})
        for ts, rpm, haul, melt, mfr, lot, shift in proc:
            p_rows.append({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "line": line,
                           "screw_rpm": round(rpm, 2), "haul_speed_mpm": round(haul, 4),
                           "melt_temp_c": round(melt, 2), "mfr": round(mfr, 3), "lot": lot})

        wall = pd.DataFrame([(ts, shift, lot, val) for ts, shift, lot, ch, val, _ in meas
                             if ch == "wall_thickness"], columns=["ts", "shift", "lot", "val"])
        pr = pd.DataFrame(proc, columns=["ts", "rpm", "haul", "melt", "mfr", "lot", "shift"])

        # one production record per shift
        shift_start = start
        while shift_start < start + timedelta(days=days):
            shift_end = shift_start + timedelta(hours=8)
            sname = shift_of(shift_start)
            planned = 480.0

            # downtime events
            events, total_down = [], 0.0
            for _ in range(rng.poisson(1.3)):
                w = [t[1] for t in DOWNTIME_TYPES]
                idx = rng.choice(len(DOWNTIME_TYPES), p=np.array(w) / sum(w))
                cat, _, (lo, hi), reasons = DOWNTIME_TYPES[idx]
                minutes = float(round(rng.uniform(lo, hi)))
                if total_down + minutes > 300:
                    continue
                total_down += minutes
                st = shift_start + timedelta(minutes=float(rng.uniform(0, planned - minutes)))
                events.append((st, st + timedelta(minutes=minutes), cat,
                               str(rng.choice(reasons)), minutes))
            for st, en, cat, reason, minutes in events:
                dt_rows.append({"start_time": st.strftime("%Y-%m-%d %H:%M:%S"),
                                "end_time": en.strftime("%Y-%m-%d %H:%M:%S"), "line": line,
                                "category": cat, "reason": reason, "minutes": minutes})

            run_min = planned - total_down
            in_shift = (pr["ts"] >= shift_start) & (pr["ts"] < shift_end)
            avg_haul = float(pr.loc[in_shift, "haul"].mean())
            total_m = run_min * avg_haul * float(rng.uniform(0.95, 1.0))

            # defects in this shift
            scrap, shift_lot = 0.0, pr.loc[in_shift, "lot"].iloc[0]
            w_in = wall[(wall["ts"] >= shift_start) & (wall["ts"] < shift_end)]
            for _, r in w_in.iterrows():
                if r["val"] < LSL or r["val"] > USL:
                    q = round(STEP_MIN * avg_haul * float(rng.uniform(0.5, 1.0)), 1)
                    scrap += q
                    def_rows.append({"timestamp": r["ts"].strftime("%Y-%m-%d %H:%M:%S"),
                                     "line": line, "product": PRODUCT,
                                     "defect_type": "Wall thickness out of spec",
                                     "quantity": q, "shift": sname, "lot": r["lot"], "notes": None})
            for dtype, rate in DEFECT_TYPES:
                lam = rate
                if dtype == "Surface scratches" and sname == "Night":
                    lam *= 1.8
                if dtype == "Marking error" and sname == "Evening":
                    lam *= 2.0
                if dtype == "Black specks":
                    lam *= lot_factor.get(shift_lot, 1.0)
                for _ in range(rng.poisson(lam)):
                    q = round(float(rng.uniform(6, 30)), 1)
                    scrap += q
                    t_def = shift_start + timedelta(minutes=float(rng.uniform(0, planned)))
                    def_rows.append({"timestamp": t_def.strftime("%Y-%m-%d %H:%M:%S"),
                                     "line": line, "product": PRODUCT, "defect_type": dtype,
                                     "quantity": q, "shift": sname, "lot": shift_lot, "notes": None})

            prod_rows.append({"start_time": shift_start.strftime("%Y-%m-%d %H:%M:%S"),
                              "end_time": shift_end.strftime("%Y-%m-%d %H:%M:%S"),
                              "line": line, "product": PRODUCT, "planned_minutes": planned,
                              "run_minutes": run_min, "ideal_rate_per_min": IDEAL_RATE,
                              "total_count": round(total_m, 1),
                              "good_count": round(max(total_m - scrap, 0), 1), "shift": sname})
            shift_start = shift_end

        # ground truth events for the AI stage (anomaly detection evaluation)
        for f_line, kind, start_day, hours, desc in FAULTS:
            if f_line == line:
                f_start = start + timedelta(days=start_day)
                ev_rows.append({"line": line, "event": kind, "start": f_start,
                                "end": f_start + timedelta(hours=hours), "description": desc})
        if bad_lot:
            ev_rows.append({"line": line, "event": "high_mfr_and_contaminated_lot",
                            "start": lots[bad_lot]["start"], "end": None,
                            "description": f"Lot {bad_lot}: MFR 0.43, contamination x4"})

    return (pd.DataFrame(m_rows), pd.DataFrame(p_rows), pd.DataFrame(prod_rows),
            pd.DataFrame(dt_rows), pd.DataFrame(def_rows), pd.DataFrame(ev_rows))


def load_demo_data(days=30, seed=42):
    """Replace all records with fresh simulated data (specs are kept)."""
    init_db()
    meas, proc, prod, down, defects, events = build_tables(days, seed)
    conn = get_connection()
    for t in ["measurements", "process_log", "production", "downtime", "defects"]:
        conn.execute(f"DELETE FROM {t}")
    meas.to_sql("measurements", conn, if_exists="append", index=False)
    proc.to_sql("process_log", conn, if_exists="append", index=False)
    prod.to_sql("production", conn, if_exists="append", index=False)
    down.to_sql("downtime", conn, if_exists="append", index=False)
    defects.to_sql("defects", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    events.to_csv(DB_PATH.parent / "sim_events.csv", index=False)
    return meas, prod, down, defects


if __name__ == "__main__":
    meas, prod, down, defects = load_demo_data()
    print("Simulated demo data loaded.")
    print(f"  measurements: {len(meas)}   production runs: {len(prod)}")
    print(f"  downtime events: {len(down)}   defect records: {len(defects)}")
    print()
    w = meas[meas["characteristic"] == "wall_thickness"]
    for line, g in w.groupby("line"):
        out = ((g["value"] < LSL) | (g["value"] > USL)).mean() * 100
        print(f"{line}: wall mean {g['value'].mean():.3f} mm, "
              f"std {g['value'].std():.3f} mm, out of spec {out:.1f}%")
