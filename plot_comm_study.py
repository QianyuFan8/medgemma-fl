# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Plot log-scale accuracy vs bytes-per-round for the RNA-seq LoRA communication study."""

from __future__ import annotations

import argparse
import json
import math
import os
from xml.sax.saxutils import escape

# Theoretical 4B bf16 full fine-tune payload for 3 clients, uplink + downlink.
DEFAULT_FULL_FT_BYTES_PER_ROUND = 3 * 2 * 4_000_000_000 * 2


def _load_results(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "experiments" in payload:
        return payload["experiments"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected a list or an object with 'experiments' in {path}")


def _nice_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    if size >= 100:
        return f"{size:.0f} {units[unit_idx]}"
    if size >= 10:
        return f"{size:.1f} {units[unit_idx]}"
    return f"{size:.2f} {units[unit_idx]}"


def write_comm_study_svg(experiments: list[dict], output_path: str, full_ft_bytes: int | None) -> None:
    points = []
    for experiment in experiments:
        bytes_per_round = experiment.get("bytes_per_round")
        accuracy = experiment.get("accuracy")
        if bytes_per_round is None or accuracy is None:
            continue
        points.append(
            {
                "name": experiment.get("name") or experiment.get("experiment") or "run",
                "bytes_per_round": float(bytes_per_round),
                "accuracy": float(accuracy),
            }
        )
    if not points:
        raise ValueError("No experiments with both bytes_per_round and accuracy were found.")

    xs = [point["bytes_per_round"] for point in points]
    if full_ft_bytes:
        xs.append(float(full_ft_bytes))
    min_x = min(xs)
    max_x = max(xs)
    log_min = math.log10(max(min_x, 1.0))
    log_max = math.log10(max(max_x, min_x * 10))
    if log_max <= log_min:
        log_max = log_min + 1.0
    pad = 0.08 * (log_max - log_min)
    log_min -= pad
    log_max += pad

    y_values = [point["accuracy"] for point in points]
    y_min = min(0.0, min(y_values) - 0.05)
    y_max = max(1.0, max(y_values) + 0.05)

    width = 920
    height = 560
    left, right, top, bottom = 90, 40, 48, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(value: float) -> float:
        return left + (math.log10(max(value, 1.0)) - log_min) / (log_max - log_min) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = ["#2E86AB", "#D1495B", "#5DA271", "#EDA65D", "#8E6C8A", "#4D908E"]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '  <title id="title">RNA-seq FL accuracy vs bytes per round</title>',
        '  <desc id="desc">Log-scale communication study for LoRA ranks and QLoRA.</desc>',
        '  <rect width="100%" height="100%" fill="#FFFDF8"/>',
        '  <text x="32" y="32" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#1F2933">'
        "TARGET RNA-seq FL: accuracy vs bytes / round"
        "</text>",
        f'  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#FFFFFF" stroke="#D8DEE9"/>',
        f'  <text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-size="13" fill="#243B53">bytes / round (log scale)</text>',
        f'  <text x="22" y="{top + plot_h / 2}" transform="rotate(-90,22,{top + plot_h / 2})" '
        'text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#243B53">Accuracy</text>',
    ]

    for tick in range(math.floor(log_min) + 1, math.ceil(log_max)):
        x = x_pos(10**tick)
        lines.append(
            f'  <line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#EEF2F7"/>'
        )
        lines.append(
            f'  <text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#486581">{_nice_bytes(10 ** tick)}</text>'
        )

    for accuracy in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_pos(accuracy)
        lines.append(
            f'  <line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#EEF2F7"/>'
        )
        lines.append(
            f'  <text x="{left - 10}" y="{y:.1f}" text-anchor="end" dominant-baseline="middle" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#486581">{accuracy:.2f}</text>'
        )

    if full_ft_bytes:
        x = x_pos(float(full_ft_bytes))
        lines.append(
            f'  <line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" '
            'stroke="#9AA5B1" stroke-dasharray="6 4"/>'
        )
        lines.append(
            f'  <text x="{x:.1f}" y="{top - 8}" text-anchor="middle" font-family="Arial, sans-serif" '
            f'font-size="11" fill="#8292A2">Full FT (theoretical {_nice_bytes(full_ft_bytes)})</text>'
        )

    for idx, point in enumerate(points):
        color = colors[idx % len(colors)]
        cx = x_pos(point["bytes_per_round"])
        cy = y_pos(point["accuracy"])
        lines.extend(
            [
                f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{color}" />',
                f'  <text x="{cx + 10:.1f}" y="{cy - 8:.1f}" font-family="Arial, sans-serif" font-size="12" '
                f'fill="#102A43">{escape(str(point["name"]))} ({point["accuracy"]:.3f})</text>',
            ]
        )

    lines.append(
        '  <text x="32" y="548" font-family="Arial, sans-serif" font-size="11" fill="#8292A2">'
        "Source: NVFlare LoRA-only adapter exchange on TARGET RNA-seq diagnosis classification. "
        "QLoRA uses 4-bit local training at rank 16."
        "</text>"
    )
    lines.append("</svg>")
    output_path = os.path.abspath(output_path)
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot RNA-seq FL communication-study results.")
    parser.add_argument(
        "--results_json",
        type=str,
        default="./comm_logs/study_results.json",
        help="JSON file written by run_comm_study.py.",
    )
    parser.add_argument(
        "--output_svg",
        type=str,
        default="./comm_logs/accuracy_vs_bytes.svg",
        help="Output SVG path.",
    )
    parser.add_argument(
        "--full_ft_bytes",
        type=int,
        default=DEFAULT_FULL_FT_BYTES_PER_ROUND,
        help="Optional theoretical Full FT x-position. Use 0 to omit.",
    )
    args = parser.parse_args()
    experiments = _load_results(args.results_json)
    full_ft_bytes = None if args.full_ft_bytes <= 0 else args.full_ft_bytes
    write_comm_study_svg(experiments, args.output_svg, full_ft_bytes)
    print(f"Wrote {os.path.abspath(args.output_svg)}")


if __name__ == "__main__":
    main()
