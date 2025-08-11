import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy image to caption directory if caption exists."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="The path of the directory for images in the dataset.",
    )
    parser.add_argument(
        "--caption_dir",
        type=str,
        required=True,
        help="The path of the directory for captions in the dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    caption_dir = args.caption_dir
    image_dir = args.image_dir

    for caption_file in os.listdir(caption_dir):
        image_file = os.path.splitext(caption_file)[0] + ".png"
        image_path = os.path.join(image_dir, image_file)
        if os.path.isfile(image_path):
            shutil.copy(os.path.join(image_dir, image_file), caption_dir)
        else:
            print(
                f"WARNING: Image file {image_file} does not exist in {image_dir}. Skipping.",
                flush=True,
            )


main()
