#!/usr/bin/env python3
"""Plot integrated depth dose (IDD) from a TOPAS cylindrical dose CSV.

For each Z bin, the script integrates dose over the transverse plane:

    IDD(z) = sum[Dose(r, phi, z) * annular_sector_area(r, phi)]

The IDD is normalized to its maximum and saved as a PNG. TOPAS bin dimensions
are read from comment lines in the CSV header. Command-line overrides are
available when those lines are absent or need to be replaced.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


BIN_HEADER = re.compile(
    r"^#\s*(R|Phi|Z)\s+in\s+(\d+)\s+bins?\s+of\s+"
    r"([0-9.eE+-]+)\s*(cm|deg)\s*$",
    re.IGNORECASE,
)


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def percentage_float(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 100:
        raise argparse.ArgumentTypeError("value must be between 0 and 100")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot normalized, area-integrated depth dose from a TOPAS "
            "cylindrical DoseToMedium CSV file."
        )
    )
    parser.add_argument("input", type=Path, help="TOPAS dose CSV file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output PNG (default: <input_stem>_idd.png)",
    )
    parser.add_argument(
        "--reverse-depth",
        action="store_true",
        help="make the highest Z-bin index the shallowest depth",
    )
    parser.add_argument(
        "--depth-offset-cm",
        type=float,
        default=0.0,
        help="offset added to every output depth (default: 0)",
    )
    parser.add_argument(
        "--z-bin-width-cm",
        type=positive_float,
        help="override the Z-bin width parsed from the TOPAS header",
    )
    parser.add_argument(
        "--r-bin-width-cm",
        type=positive_float,
        help="override the R-bin width parsed from the TOPAS header",
    )
    parser.add_argument(
        "--phi-bin-width-deg",
        type=positive_float,
        help="override the Phi-bin width parsed from the TOPAS header",
    )
    parser.add_argument(
        "--r-min-cm",
        type=float,
        default=0.0,
        help="inner radius of radial bin zero (default: 0)",
    )
    parser.add_argument(
        "--smoothing-window-cm",
        type=positive_float,
        default=1.0,
        help="width of the local quadratic smoothing window (default: 1.0 cm)",
    )
    parser.add_argument(
        "--analysis-min-percent",
        type=percentage_float,
        default=10.0,
        help=(
            "exclude residuals where the smoothed IDD is below this percentage "
            "of its maximum (default: 10)"
        ),
    )
    return parser.parse_args()


def read_topas_csv(
    path: Path,
) -> tuple[dict[str, tuple[int, float]], list[tuple[int, int, int, float]]]:
    dimensions: dict[str, tuple[int, float]] = {}
    rows: list[tuple[int, int, int, float]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                match = BIN_HEADER.match(stripped)
                if match:
                    axis, count, width, _unit = match.groups()
                    dimensions[axis.lower()] = (int(count), float(width))
                continue

            fields = next(csv.reader([raw_line], skipinitialspace=True))
            if len(fields) < 4:
                raise ValueError(
                    f"{path}:{line_number}: expected R, Phi, Z, and dose columns"
                )
            try:
                r_bin = int(fields[0])
                phi_bin = int(fields[1])
                z_bin = int(fields[2])
                dose = float(fields[3])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid numeric TOPAS row"
                ) from error

            if min(r_bin, phi_bin, z_bin) < 0:
                raise ValueError(
                    f"{path}:{line_number}: bin indices cannot be negative"
                )
            rows.append((r_bin, phi_bin, z_bin, dose))

    if not rows:
        raise ValueError(f"{path}: no TOPAS dose rows found")
    return dimensions, rows


def require_dimension(
    override: float | None,
    dimensions: dict[str, tuple[int, float]],
    axis: str,
    option: str,
) -> float:
    if override is not None:
        return override
    if axis not in dimensions:
        raise ValueError(
            f"missing {axis.upper()}-bin width in TOPAS header; provide {option}"
        )
    return dimensions[axis][1]


def calculate_idd(
    rows: list[tuple[int, int, int, float]],
    r_width_cm: float,
    phi_width_deg: float,
    r_min_cm: float,
) -> dict[int, float]:
    if r_min_cm < 0:
        raise ValueError("--r-min-cm cannot be negative")

    phi_width_rad = math.radians(phi_width_deg)
    idd_by_z: dict[int, float] = {}
    for r_bin, _phi_bin, z_bin, dose_gy in rows:
        r_inner = r_min_cm + r_bin * r_width_cm
        r_outer = r_inner + r_width_cm
        sector_area_cm2 = 0.5 * (r_outer**2 - r_inner**2) * phi_width_rad
        idd_by_z[z_bin] = idd_by_z.get(z_bin, 0.0) + dose_gy * sector_area_cm2
    return idd_by_z


def build_curve(
    idd_by_z: dict[int, float],
    z_bin_count: int,
    z_width_cm: float,
    depth_offset_cm: float,
    reverse_depth: bool,
) -> tuple[list[float], list[float]]:
    peak = max(idd_by_z.values())
    if peak <= 0:
        raise ValueError("IDD values do not contain a positive peak")

    depths: list[float] = []
    normalized_idd: list[float] = []
    for z_bin in sorted(idd_by_z, reverse=reverse_depth):
        depth_bin = z_bin_count - 1 - z_bin if reverse_depth else z_bin
        depths.append(depth_offset_cm + (depth_bin + 0.5) * z_width_cm)
        normalized_idd.append(100.0 * idd_by_z[z_bin] / peak)
    return depths, normalized_idd


def require_analysis_dependencies() -> None:
    try:
        import numpy  # noqa: F401
        import matplotlib
    except ModuleNotFoundError as error:
        raise SystemExit(
            "NumPy and Matplotlib are required. Install them with: "
            "python3 -m pip install numpy matplotlib"
        ) from error

    matplotlib.use("Agg")


def smooth_local_quadratic(
    depths: list[float],
    values: list[float],
    window_cm: float,
) -> tuple[object, int]:
    """Return a local-quadratic baseline and the odd window size in bins."""
    import numpy as np

    x = np.asarray(depths, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.size != y.size:
        raise ValueError("depth and IDD arrays must have the same length")
    if x.size < 5:
        raise ValueError("at least five depth bins are required for smoothing")

    spacings = np.diff(x)
    if np.any(spacings <= 0):
        raise ValueError("depth values must be strictly increasing")
    bin_width_cm = float(np.median(spacings))

    window_bins = max(5, int(round(window_cm / bin_width_cm)))
    if window_bins % 2 == 0:
        window_bins += 1
    largest_odd_window = x.size if x.size % 2 == 1 else x.size - 1
    window_bins = min(window_bins, largest_odd_window)

    half_window = window_bins // 2
    smoothed = np.empty_like(y)
    for index in range(x.size):
        start = max(0, index - half_window)
        end = min(x.size, start + window_bins)
        start = max(0, end - window_bins)

        local_x = x[start:end] - x[index]
        local_y = y[start:end]
        coefficients = np.polyfit(local_x, local_y, deg=2)
        smoothed[index] = np.polyval(coefficients, 0.0)

    # A dose baseline is physical only when non-negative.
    return np.clip(smoothed, 0.0, None), window_bins


def calculate_fluctuation(
    raw_idd: list[float],
    smoothed_idd: object,
    analysis_min_percent: float,
) -> tuple[object, dict[str, float | int]]:
    """Calculate percent residuals and summary metrics in the analysis region."""
    import numpy as np

    raw = np.asarray(raw_idd, dtype=float)
    smooth = np.asarray(smoothed_idd, dtype=float)
    if raw.shape != smooth.shape:
        raise ValueError("raw and smoothed IDD arrays must have the same shape")

    smooth_peak = float(np.max(smooth))
    if smooth_peak <= 0:
        raise ValueError("smoothed IDD does not contain a positive value")

    minimum = analysis_min_percent / 100.0 * smooth_peak
    mask = np.isfinite(raw) & np.isfinite(smooth) & (smooth > 0) & (smooth >= minimum)
    if not np.any(mask):
        raise ValueError("no depth bins remain in the fluctuation analysis region")

    residual = np.full(raw.shape, np.nan, dtype=float)
    residual[mask] = 100.0 * (raw[mask] - smooth[mask]) / smooth[mask]
    analyzed = residual[mask]
    absolute = np.abs(analyzed)
    metrics: dict[str, float | int] = {
        "rms_percent": float(np.sqrt(np.mean(np.square(analyzed)))),
        "p95_absolute_percent": float(np.percentile(absolute, 95)),
        "max_absolute_percent": float(np.max(absolute)),
        "analyzed_bins": int(analyzed.size),
    }
    return residual, metrics


def plot_idd(
    path: Path,
    depths: list[float],
    normalized_idd: list[float],
    smoothed_idd: object,
    fluctuation: object,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    figure, (idd_axis, fluctuation_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.08},
    )

    idd_axis.plot(
        depths,
        normalized_idd,
        color="#185699",
        linewidth=1.2,
        label="Raw IDD",
    )
    idd_axis.plot(
        depths,
        smoothed_idd,
        color="#d95f02",
        linewidth=2.0,
        label="Smoothed mean",
    )
    idd_axis.set_title("Integrated Depth Dose (IDD)", fontsize=16, pad=14)
    idd_axis.set_ylabel("Normalized IDD (%)", fontsize=12)
    idd_axis.set_xlim(min(depths), max(depths))
    idd_axis.set_ylim(0, max(105.0, float(np.max(smoothed_idd)) * 1.05))
    idd_axis.grid(True, color="#dce2e9", linewidth=0.8)
    idd_axis.legend(loc="best", frameon=False)

    fluctuation_axis.axhspan(-1, 1, color="#2ca25f", alpha=0.16)
    fluctuation_axis.axhspan(1, 2, color="#f0ad4e", alpha=0.14)
    fluctuation_axis.axhspan(-2, -1, color="#f0ad4e", alpha=0.14)
    fluctuation_axis.axhline(0, color="#4a5568", linewidth=1.0)
    fluctuation_axis.axhline(1, color="#2ca25f", linestyle=":", linewidth=1.0)
    fluctuation_axis.axhline(-1, color="#2ca25f", linestyle=":", linewidth=1.0)
    fluctuation_axis.axhline(2, color="#d98e04", linestyle="--", linewidth=1.0)
    fluctuation_axis.axhline(-2, color="#d98e04", linestyle="--", linewidth=1.0)
    fluctuation_axis.plot(
        depths,
        fluctuation,
        color="#6a3d9a",
        linewidth=1.0,
    )

    finite_fluctuation = np.asarray(fluctuation)[np.isfinite(fluctuation)]
    residual_limit = max(3.0, float(np.max(np.abs(finite_fluctuation))) * 1.05)
    fluctuation_axis.set_ylim(-residual_limit, residual_limit)
    fluctuation_axis.set_xlabel("Depth (cm)", fontsize=12)
    fluctuation_axis.set_ylabel("Fluctuation (%)", fontsize=11)
    fluctuation_axis.grid(True, axis="x", color="#dce2e9", linewidth=0.8)

    for axis in (idd_axis, fluctuation_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(path, format="png", dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    require_analysis_dependencies()
    dimensions, rows = read_topas_csv(args.input)

    z_width_cm = require_dimension(
        args.z_bin_width_cm, dimensions, "z", "--z-bin-width-cm"
    )
    r_width_cm = require_dimension(
        args.r_bin_width_cm, dimensions, "r", "--r-bin-width-cm"
    )
    phi_width_deg = require_dimension(
        args.phi_bin_width_deg, dimensions, "phi", "--phi-bin-width-deg"
    )

    header_z_count = dimensions.get("z", (0, 0.0))[0]
    observed_z_count = max(row[2] for row in rows) + 1
    z_bin_count = max(header_z_count, observed_z_count)

    idd_by_z = calculate_idd(rows, r_width_cm, phi_width_deg, args.r_min_cm)
    depths, normalized_idd = build_curve(
        idd_by_z,
        z_bin_count,
        z_width_cm,
        args.depth_offset_cm,
        args.reverse_depth,
    )
    smoothed_idd, smoothing_window_bins = smooth_local_quadratic(
        depths,
        normalized_idd,
        args.smoothing_window_cm,
    )
    fluctuation, metrics = calculate_fluctuation(
        normalized_idd,
        smoothed_idd,
        args.analysis_min_percent,
    )

    output = args.output or args.input.with_name(f"{args.input.stem}_idd.png")
    if output.suffix.lower() != ".png":
        raise ValueError("output filename must end in .png")
    plot_idd(output, depths, normalized_idd, smoothed_idd, fluctuation)

    orientation = "reversed Z-bin order" if args.reverse_depth else "CSV Z-bin order"
    print(f"Plotted {len(depths)} depth bins to {output} ({orientation}).")
    print(
        f"Local quadratic mean: {args.smoothing_window_cm:g} cm "
        f"({smoothing_window_bins} bins)."
    )
    print(
        f"Fluctuation diagnostics above {args.analysis_min_percent:g}% of the "
        f"smoothed maximum: RMS={metrics['rms_percent']:.3f}%, "
        f"P95(abs)={metrics['p95_absolute_percent']:.3f}%, "
        f"Max(abs)={metrics['max_absolute_percent']:.3f}%, "
        f"bins={metrics['analyzed_bins']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
