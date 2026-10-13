"""Print the defect Pareto, scrap rates, defect mix by shift, and unusual lots."""
from db import read_table
from metrics.defects import pareto, scrap_rate, defect_mix, lot_summary, lot_defect_outliers


def main():
    defects = read_table("defects")
    prod = read_table("production")
    log = read_table("process_log")

    print(f"{len(defects)} defect records, {defects['quantity'].sum():.0f} m scrapped\n")

    print("Pareto by defect type (vital few = causes needed to reach 80%)")
    print(pareto(defects).to_string(index=False))

    print("\nScrap rate by line")
    print(scrap_rate(defects, prod, "line").to_string(index=False))
    print("\nScrap rate by shift")
    print(scrap_rate(defects, prod, "shift").to_string(index=False))

    mix = defect_mix(defects, "defect_type", "shift")
    print("\nDefect events by type and shift")
    print(mix["table"].to_string())
    verdict = "the mix of causes differs between shifts" if mix["different"] \
        else "no clear difference between shifts"
    print(f"Chi-square p-value {mix['p_value']:.4f}, effect size (Cramer's V) "
          f"{mix['cramers_v']}: {verdict}")

    lots = lot_summary(defects, log)
    print("\nResin lots with the highest scrap per 100 running hours")
    print(lots.head(5).to_string(index=False))
    flagged = lots[lots["flag"]]
    print("\nFlagged lots:", ", ".join(flagged["lot"]) if len(flagged) else "none")

    odd = lot_defect_outliers(defects, log)
    print("\nLots unusually bad for one specific defect type")
    print(odd.to_string(index=False) if len(odd) else "none")


if __name__ == "__main__":
    main()
