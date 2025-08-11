import argparse

from tile_image import tile_all_images_in_directory


def parse_args():
    parser = argparse.ArgumentParser(description="Tile images into smaller sections.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The path of the input directory containing images to be tiled.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path of the directory for tiles to be saved in.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="The size of the tiles.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tile_all_images_in_directory(args.input_dir, args.output_dir, args.tile_size)


main()
