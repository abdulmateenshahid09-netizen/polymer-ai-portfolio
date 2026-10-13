"""Process capability: Cp, Cpk, Pp, Ppk. Python does the maths, never the LLM."""
import numpy as np
from scipy.stats import norm, skew

D2 = 1.128   # constant for moving ranges of 2 readings (individuals chart)
MIN_READINGS = 30


def rating(index):
    """Plain-language reading of a Cpk or Ppk value."""
    if index is None:
        return "not available"
    if index >= 1.67:
        return "excellent"
    if index >= 1.33:
        return "capable"
    if index >= 1.00:
        return "marginal"
    return "not capable"


def capability(values, lsl=None, usl=None):
    """Capability indices for a series of individual readings in time order.

    Cp and Cpk use the short-term (within) spread, from the average moving range.
    Pp and Ppk use the overall standard deviation.
    Give lsl, usl or both. With one limit only, Cp and Pp are not defined.
    """
    if lsl is None and usl is None:
        raise ValueError("Give at least one specification limit (lsl or usl)")
    if lsl is not None and usl is not None and lsl >= usl:
        raise ValueError("lsl must be smaller than usl")

    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        raise ValueError("Need at least 2 readings")

    mean = float(x.mean())
    sigma_overall = float(x.std(ddof=1))
    sigma_within = float(np.abs(np.diff(x)).mean() / D2)
    if sigma_within == 0 or sigma_overall == 0:
        raise ValueError("All readings are identical, so capability cannot be calculated")

    def indices(sigma):
        upper = (usl - mean) / (3 * sigma) if usl is not None else None
        lower = (mean - lsl) / (3 * sigma) if lsl is not None else None
        both = [v for v in (upper, lower) if v is not None]
        p = (usl - lsl) / (6 * sigma) if (usl is not None and lsl is not None) else None
        return p, min(both)

    cp, cpk = indices(sigma_within)
    pp, ppk = indices(sigma_overall)

    below = float(np.mean(x < lsl)) * 100 if lsl is not None else 0.0
    above = float(np.mean(x > usl)) * 100 if usl is not None else 0.0
    ppm = 0.0
    if lsl is not None:
        ppm += norm.cdf((lsl - mean) / sigma_overall) * 1e6
    if usl is not None:
        ppm += norm.sf((usl - mean) / sigma_overall) * 1e6

    warnings = []
    if len(x) < MIN_READINGS:
        warnings.append(f"Only {len(x)} readings (fewer than {MIN_READINGS}): "
                        "treat the indices as indicative only.")
    if ppk < 0.8 * cpk:
        warnings.append("Ppk is much lower than Cpk: the process mean drifts or shifts over "
                        "time, so short-term capability overstates what the customer receives.")
    if len(x) >= 8 and abs(float(skew(x))) > 1:
        warnings.append("The readings are clearly skewed, so the normal-distribution "
                        "estimates (PPM) may be unreliable.")

    def r(v):
        return None if v is None else round(float(v), 3)

    return {
        "n": int(len(x)), "mean": round(mean, 4),
        "sigma_within": round(sigma_within, 4), "sigma_overall": round(sigma_overall, 4),
        "cp": r(cp), "cpk": r(cpk), "pp": r(pp), "ppk": r(ppk),
        "cpk_rating": rating(r(cpk)), "ppk_rating": rating(r(ppk)),
        "percent_below_lsl": round(below, 2), "percent_above_usl": round(above, 2),
        "expected_ppm_out": round(float(ppm), 1),
        "warnings": warnings,
    }
