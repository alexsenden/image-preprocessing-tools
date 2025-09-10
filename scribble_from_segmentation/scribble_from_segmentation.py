import argparse
import os
import numpy as np
import cv2

from PIL import Image
from skimage.morphology import skeletonize

IMAGE_FILE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tif",
)

BORDER_PADDING = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate scribble annotations from segmentation masks."
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
        required=True,
        help="The path of the directory for tiles to be saved in.",
    )
    return parser.parse_args()


def discover_segmentation_classes(image):
    return [list(colour) for colour in list(set(image.getdata()))]


def set_colour(colour_image, binary_image, colour):
    np.copyto(colour_image, colour, where=binary_image[..., None])


def generate_comparison(image1, image2):
    height, width = image1.shape

    blue_arr = np.zeros((height, width, 3), dtype=np.uint8)
    red_arr = np.zeros((height, width, 3), dtype=np.uint8)

    blue = np.array([0, 0, 255], dtype=blue_arr.dtype)
    red = np.array([255, 0, 0], dtype=red_arr.dtype)

    set_colour(blue_arr, image1, blue)
    set_colour(red_arr, image2, red)

    return blue_arr + red_arr


def main():
    args = parse_args()

    images = [
        file
        for file in os.listdir(args.input_dir)
        if file.endswith(IMAGE_FILE_EXTENSIONS)
    ]
    if len(images) == 0:
        print(f"ERROR: No files discovered in input directory {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    for image_file_name in images:
        image_file_path = os.path.join(args.input_dir, image_file_name)
        print(f"INFO: Processing image {image_file_path}")

        pil_image = Image.open(image_file_path)
        segmentation_classes = discover_segmentation_classes(pil_image)

        np_image = np.asarray(pil_image)
        np_image = np.pad(
            np_image,
            (
                (BORDER_PADDING, BORDER_PADDING),
                (BORDER_PADDING, BORDER_PADDING),
                (0, 0),
            ),
            mode="edge",
        )

        scribble_annotation = np.zeros(
            (np_image.shape[0], np_image.shape[1], 3), dtype=np.uint8
        )

        for segmentation_class in segmentation_classes:
            boolean_image = np.all(np_image == segmentation_class, axis=-1).astype(
                np.uint8
            )

            # Check if class is background
            if np.sum(boolean_image) / boolean_image.size > 0.5:
                print(
                    f"INFO: Class {segmentation_class} detected as background class. Skipping."
                )
                continue

            kernel_size = 11
            dilation_iterations = 5

            ellipse_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )

            row_kernel = np.ones((kernel_size, 3), dtype=np.uint8)
            thin_row_kernel = np.ones((7, 1), dtype=np.uint8)
            dilated = cv2.dilate(
                boolean_image, row_kernel, iterations=dilation_iterations
            )

            skeleton = skeletonize(dilated).astype(np.uint8)

            denubbed = cv2.dilate(skeleton, ellipse_kernel, iterations=2)
            denubbed = cv2.morphologyEx(
                denubbed, cv2.MORPH_OPEN, row_kernel, iterations=4
            )

            regrown = cv2.dilate(denubbed, ellipse_kernel, iterations=1)
            regrown_contracted = cv2.erode(regrown, thin_row_kernel, iterations=8)

            skele2 = skeletonize(regrown_contracted).astype(np.uint8)
            regrown2 = cv2.dilate(skele2, ellipse_kernel, iterations=1)

            set_colour(
                colour_image=scribble_annotation,
                binary_image=regrown2.astype(bool),
                colour=np.asarray(segmentation_class[:3], dtype=np.uint8),
            )

        Image.fromarray(
            scribble_annotation[
                BORDER_PADDING:-BORDER_PADDING, BORDER_PADDING:-BORDER_PADDING
            ]
        ).save(
            os.path.join(
                args.output_dir,
                f"{os.path.splitext(image_file_name)[0]}_scribble.png",
            )
        )


main()
