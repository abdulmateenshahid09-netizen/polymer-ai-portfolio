from db import get_connection, init_db

PRODUCT = "PE100 SDR11 110mm"

SPECS = [
    # product, characteristic, LSL, USL, target, unit
    (PRODUCT, "wall_thickness", 10.0, 11.1, 10.4, "mm"),
    (PRODUCT, "outside_diameter", 110.0, 110.3, 110.15, "mm"),
]


def add_specs():
    init_db()
    conn = get_connection()
    conn.executemany(
        "INSERT OR REPLACE INTO specs (product, characteristic, lsl, usl, target, unit) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        SPECS,
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    add_specs()
    print("Specs saved:")
    for s in SPECS:
        print(f"  {s[1]:18s} LSL={s[2]}  USL={s[3]}  target={s[4]} {s[5]}")