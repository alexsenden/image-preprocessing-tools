import argparse
import os

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="A simple tool for recolouring segmentation masks."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The path of the input directory containing a segmentation mask to be reduced.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="The path of the directory for tiles to be saved in. If not provided, the images will be overwritten.",
    )
    parser.add_argument(
        "--inplace",
        type=str,
        required=False,
        default=False,
        help=("Whether to recolour the images in place (i.e. overwrite the original)"),
    )
    parser.add_argument(
        "--colour_map",
        type=str,
        required=True,
        help=(
            "A mapping of original colours to new colours in the format "
            "'r1,g1,b1:r2,g2,b2;r3,g3,b3:r4,g4,b4' where 'r1,g1,b1' is the "
            "original colour and 'r2,g2,b2' is the new colour."
        ),
    )

    args = parser.parse_args()

    if args.output_dir is None and not args.inplace:
        parser.error(
            "Either --output_dir must be specified or --inplace must be set to True."
        )

    return args


def parse_colour_map(colour_map_str: str) -> dict:
    colour_map = {}
    mappings = colour_map_str.split(";")

    for mapping in mappings:
        original, new = mapping.split(":")

        original_colour = tuple(map(int, original.split(",")))
        new_colour = tuple(map(int, new.split(",")))

        colour_map[original_colour] = new_colour

    return colour_map


def recolour_image(image_path: str, output_path: str, colour_map: dict):
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    for original_colour, new_colour in colour_map.items():
        mask = np.all(image_array == original_colour, axis=-1)
        image_array[mask] = new_colour

    recoloured_image = Image.fromarray(image_array.astype(np.uint8))
    recoloured_image.save(output_path)


def recolour_images_in_directory_recursive(
    input_dir: str, output_dir: str, colour_map: dict
):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        ):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            recolour_image(input_path, output_path, colour_map)
        elif os.path.isdir(os.path.join(input_dir, filename)):
            sub_input_dir = os.path.join(input_dir, filename)
            sub_output_dir = os.path.join(output_dir, filename)

            os.makedirs(sub_output_dir, exist_ok=True)
            recolour_images_in_directory_recursive(
                sub_input_dir, sub_output_dir, colour_map
            )


if __name__ == "__main__":
    args = parse_args()
    colour_map = parse_colour_map(args.colour_map)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    recolour_images_in_directory_recursive(args.input_dir, args.output_dir, colour_map)
