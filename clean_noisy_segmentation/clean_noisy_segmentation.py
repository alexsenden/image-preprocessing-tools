import argparse
import os

import numpy as np

from PIL import Image
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign each pixel to the nearest colour in a provided palette."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input directory containing segmentation images (recursively processed).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help=(
            "Directory where cleaned images will be written. If not provided, must use --inplace."
        ),
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input images in place (if set, --output_dir may be omitted).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--colours",
        type=str,
        help=("Semicolon-separated list of RGB triples, e.g. '0,0,0;255,0,0;0,255,0'"),
    )
    group.add_argument(
        "--colours_file",
        type=str,
        help=("Path to a file listing one RGB triple per line, e.g. 'r,g,b'."),
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".png,.jpg,.jpeg,.tiff,.bmp,.gif",
        help="Comma-separated list of allowed image extensions (default includes common types).",
    )
    parser.add_argument(
        "--name_filter",
        type=str,
        default="",
        help="Only process files whose name contains this substring (default: empty string matches all).",
    )

    return parser.parse_args()


def parse_palette_from_string(colours_str: str) -> np.ndarray:
    parts = [p.strip() for p in colours_str.split(";") if p.strip()]
    pal = []
    for p in parts:
        rgb = tuple(int(x) for x in p.split(","))
        if len(rgb) != 3:
            raise ValueError(f"Invalid colour triple: {p}")
        pal.append(rgb)
    return np.asarray(pal, dtype=np.uint8)


def parse_palette_from_file(path: str) -> np.ndarray:
    pal = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rgb = tuple(int(x) for x in line.split(","))
            if len(rgb) != 3:
                raise ValueError(f"Invalid colour triple in file {path}: {line}")
            pal.append(rgb)
    if not pal:
        raise ValueError(f"No colours found in file: {path}")
    return np.asarray(pal, dtype=np.uint8)


def nearest_palette_image(image_array: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Assign each pixel in `image_array` (H,W,3 uint8) to the nearest colour in `palette` (K,3 uint8).
    Returns recoloured image array with same shape and dtype uint8.
    """
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("image_array must have shape (H, W, 3)")

    h, w, _ = image_array.shape
    flat = image_array.reshape(-1, 3).astype(np.int16)
    pal = palette.astype(np.int16)

    # Compute squared distances between each pixel and each palette colour.
    # distances shape: (N_pixels, K)
    d = np.sum((flat[:, None, :] - pal[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d, axis=1)
    new_flat = pal[idx]
    new = new_flat.reshape(h, w, 3).astype(np.uint8)
    return new


def process_file(input_path: str, output_path: str, palette: np.ndarray):
    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"Skipping {input_path}: cannot open image ({e})")
        return

    arr = np.array(img, dtype=np.uint8)
    cleaned = nearest_palette_image(arr, palette)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(cleaned).save(output_path)


def process_directory(
    input_dir: str, output_dir: str, palette: np.ndarray, exts: List[str], inplace: bool, name_filter: str = ""
):
    exts = [e.lower().strip() for e in exts]
    for root, dirs, files in os.walk(input_dir):
        # Determine the corresponding output root
        rel = os.path.relpath(root, input_dir)
        out_root = os.path.join(output_dir, rel) if not inplace else root
        os.makedirs(out_root, exist_ok=True)

        for fname in files:
            if not any(fname.lower().endswith(e) for e in exts):
                continue
            if name_filter and name_filter not in fname:
                continue
            in_path = os.path.join(root, fname)
            out_path = os.path.join(out_root, fname)
            process_file(in_path, out_path, palette)


def main():
    args = parse_args()

    if args.inplace:
        out_dir = args.input_dir
    else:
        if args.output_dir is None:
            raise SystemExit("Either --output_dir or --inplace must be specified")
        out_dir = args.output_dir

    if args.colours_file:
        palette = parse_palette_from_file(args.colours_file)
    else:
        palette = parse_palette_from_string(args.colours)

    exts = [e if e.startswith(".") else "." + e for e in args.exts.split(",")]

    if not args.inplace:
        os.makedirs(out_dir, exist_ok=True)

    print(
        f"Processing: input={args.input_dir} -> output={out_dir}, colours={len(palette)}"
    )
    process_directory(args.input_dir, out_dir, palette, exts, args.inplace, args.name_filter)
    print("Done.")


if __name__ == "__main__":
    main()
