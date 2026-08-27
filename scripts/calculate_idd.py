#!/usr/bin/env python3
"""Plot integrated depth dose (IDD) from a TOPAS cylindrical dose CSV.

For each Z bin, the script integrates dose over the transverse plane:

    IDD(z) = sum[Dose(r, phi, z) * annular_sector_area(r, phi)]

The IDD is normalized to its peak and written directly to a PNG plot. TOPAS bin
dimensions are read from comment lines in the CSV header. Command-line
overrides are available when those lines are absent or need to be replaced.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import struct
import zlib
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
    return parser.parse_args()


def read_topas_csv(path: Path) -> tuple[dict[str, tuple[int, float]], list[tuple[int, int, int, float]]]:
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
                raise ValueError(f"{path}:{line_number}: bin indices cannot be negative")
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


FONT = {
    " ": ("00000",) * 7,
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    "(": ("00110", "01100", "01000", "01000", "01000", "01100", "00110"),
    ")": ("01100", "00110", "00010", "00010", "00010", "00110", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.rows = [bytearray([255, 255, 255]) * width for _ in range(height)]

    def pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = x * 3
            self.rows[y][offset : offset + 3] = bytes(color)

    def rectangle(
        self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]
    ) -> None:
        for y in range(max(0, y0), min(self.height, y1 + 1)):
            for x in range(max(0, x0), min(self.width, x1 + 1)):
                self.pixel(x, y, color)

    def line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        radius = max(0, width // 2)
        while True:
            self.rectangle(x0 - radius, y0 - radius, x0 + radius, y0 + radius, color)
            if x0 == x1 and y0 == y1:
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    def text_width(self, text: str, scale: int) -> int:
        return max(0, len(text) * 6 * scale - scale)

    def text(
        self,
        x: int,
        y: int,
        text: str,
        color: tuple[int, int, int],
        scale: int = 2,
    ) -> None:
        cursor = x
        for character in text.upper():
            glyph = FONT.get(character, FONT[" "])
            for row_index, row in enumerate(glyph):
                for column_index, bit in enumerate(row):
                    if bit == "1":
                        self.rectangle(
                            cursor + column_index * scale,
                            y + row_index * scale,
                            cursor + (column_index + 1) * scale - 1,
                            y + (row_index + 1) * scale - 1,
                            color,
                        )
            cursor += 6 * scale

    def save_png(self, path: Path) -> None:
        def chunk(kind: bytes, payload: bytes) -> bytes:
            checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

        raw = b"".join(b"\x00" + bytes(row) for row in self.rows)
        png = b"\x89PNG\r\n\x1a\n"
        png += chunk("IHDR".encode(), struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        png += chunk("IDAT".encode(), zlib.compress(raw, level=9))
        png += chunk("IEND".encode(), b"")
        path.write_bytes(png)


def plot_idd(
    path: Path,
    idd_by_z: dict[int, float],
    z_bin_count: int,
    z_width_cm: float,
    depth_offset_cm: float,
    reverse_depth: bool,
) -> None:
    peak = max(idd_by_z.values())
    if peak <= 0:
        raise ValueError("IDD values do not contain a positive peak")
    path.parent.mkdir(parents=True, exist_ok=True)

    curve: list[tuple[float, float]] = []
    for z_bin in sorted(idd_by_z, reverse=reverse_depth):
        depth_bin = z_bin_count - 1 - z_bin if reverse_depth else z_bin
        depth_cm = depth_offset_cm + (depth_bin + 0.5) * z_width_cm
        curve.append((depth_cm, 100.0 * idd_by_z[z_bin] / peak))

    width, height = 1200, 750
    left, right, top, bottom = 105, 45, 100, 95
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    min_depth = min(point[0] for point in curve)
    max_depth = max(point[0] for point in curve)
    if math.isclose(min_depth, max_depth):
        max_depth = min_depth + 1.0

    canvas = Canvas(width, height)
    grid = (220, 226, 233)
    axis = (45, 55, 72)
    ink = (24, 86, 153)

    for tick in range(6):
        fraction = tick / 5
        y = round(plot_bottom - fraction * plot_height)
        canvas.line(plot_left, y, plot_right, y, grid)
        label = f"{round(fraction * 100):d}"
        canvas.text(plot_left - 18 - canvas.text_width(label, 2), y - 7, label, axis, 2)

    for tick in range(9):
        fraction = tick / 8
        x = round(plot_left + fraction * plot_width)
        canvas.line(x, plot_top, x, plot_bottom, grid)
        value = min_depth + fraction * (max_depth - min_depth)
        label = f"{value:.1f}"
        canvas.text(x - canvas.text_width(label, 2) // 2, plot_bottom + 18, label, axis, 2)

    canvas.line(plot_left, plot_top, plot_left, plot_bottom, axis, 3)
    canvas.line(plot_left, plot_bottom, plot_right, plot_bottom, axis, 3)

    title = "INTEGRATED DEPTH DOSE (IDD)"
    canvas.text((width - canvas.text_width(title, 3)) // 2, 30, title, axis, 3)
    y_label = "NORMALIZED IDD (%)"
    canvas.text(plot_left, 72, y_label, axis, 2)
    x_label = "DEPTH (CM)"
    canvas.text((width - canvas.text_width(x_label, 2)) // 2, height - 42, x_label, axis, 2)

    def coordinates(point: tuple[float, float]) -> tuple[int, int]:
        depth, normalized = point
        x = plot_left + (depth - min_depth) / (max_depth - min_depth) * plot_width
        y = plot_bottom - normalized / 100.0 * plot_height
        return round(x), round(y)

    pixel_curve = [coordinates(point) for point in curve]
    for first, second in zip(pixel_curve, pixel_curve[1:]):
        canvas.line(*first, *second, ink, 3)

    canvas.save_png(path)


def main() -> int:
    args = parse_args()
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
    output = args.output or args.input.with_name(f"{args.input.stem}_idd.png")
    if output.suffix.lower() != ".png":
        raise ValueError("output filename must end in .png")
    plot_idd(
        output,
        idd_by_z,
        z_bin_count,
        z_width_cm,
        args.depth_offset_cm,
        args.reverse_depth,
    )

    orientation = "reversed Z-bin order" if args.reverse_depth else "CSV Z-bin order"
    print(f"Plotted {len(idd_by_z)} depth bins to {output} ({orientation}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
