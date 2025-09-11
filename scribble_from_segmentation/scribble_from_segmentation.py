import argparse
import cv2
import math
import numpy as np
import os

from PIL import Image
from skimage import measure, morphology

IMAGE_FILE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tif",
)

BORDER_PADDING = 200
KERNEL_SIZE = 11
SMALL_KERNEL_SIZE = 7
DILATION_ITERATIONS = 5
BACKGROUND_THRESHOLD = 0.5

ELLIPSE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
)
SMALL_ELLIPSE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (SMALL_KERNEL_SIZE, SMALL_KERNEL_SIZE)
)


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


def save_image(image, path):
    Image.fromarray(image.astype(np.uint8)).save(path)


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


def get_image_paths(input_dir):
    images = [
        file for file in os.listdir(input_dir) if file.endswith(IMAGE_FILE_EXTENSIONS)
    ]

    # Ensure at least one image exists
    if len(images) == 0:
        raise Exception(f"ERROR: No files discovered in input directory {input_dir}")

    return images


def get_row_direction(boolean_image):
    # Preprocess: dilate image to merge nearby objects
    dilated = cv2.dilate(np.transpose(boolean_image), ELLIPSE_KERNEL, iterations=2)

    # Get largest object from dilated image
    labeled = measure.label(dilated, connectivity=2)
    regions = measure.regionprops(labeled)
    largest_region = max(regions, key=lambda r: r.area)

    # Align to axis if near one
    theta = largest_region.orientation
    if -0.1 < theta and theta < 0.1:
        theta = 0
    elif theta < -1.5 or theta > 1.5:
        theta = math.pi / 2.0

    return theta


def oriented_line_kernel(theta, size, thickness):
    kernel = np.zeros((size, size), dtype=np.uint8)

    # Center point (size should always be odd)
    center = (size - 1) // 2

    # Direction vector
    dx = np.round(np.cos(theta), 3)
    dy = np.round(np.sin(theta), 3)

    # Large enough line endpoints in both directions
    x0 = int(center - dx * size)
    y0 = int(center - dy * size)
    x1 = int(center + dx * size)
    y1 = int(center + dy * size)

    # Draw line on kernel
    cv2.line(kernel, (x0, y0), (x1, y1), 1, thickness=thickness)

    return kernel


def get_convolutional_kernels(image):
    row_direction_theta = get_row_direction(
        image[BORDER_PADDING:-BORDER_PADDING, BORDER_PADDING:-BORDER_PADDING]
    )

    thin_kernel = oriented_line_kernel(row_direction_theta, SMALL_KERNEL_SIZE, 1)
    thick_kernel = oriented_line_kernel(row_direction_theta, KERNEL_SIZE, 2)

    return thin_kernel, thick_kernel


def generate_scribble_for_boolean(boolean_image):
    ELLIPSE_KERNEL = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
    )
    thin_row_kernel, row_kernel = get_convolutional_kernels(boolean_image)

    # Dilate mask to morph shapes together, produces less nubs in skeleton
    dilated = cv2.dilate(boolean_image, row_kernel, iterations=DILATION_ITERATIONS)
    skeleton = morphology.skeletonize(dilated).astype(np.uint8)

    # Remove some extra nubs via repeated opening
    denubbed = cv2.dilate(skeleton, ELLIPSE_KERNEL, iterations=2)
    denubbed = cv2.morphologyEx(denubbed, cv2.MORPH_OPEN, row_kernel, iterations=4)
    regrown = cv2.dilate(denubbed, ELLIPSE_KERNEL, iterations=1)

    # Contract vertically, since row kernel overexpands vertically
    regrown_contracted = cv2.erode(regrown, thin_row_kernel, iterations=8)

    # Remorphology.skeletonize and dilate to smoothen and unify line width
    second_skeleton = morphology.skeletonize(regrown_contracted).astype(np.uint8)
    return cv2.dilate(second_skeleton, ELLIPSE_KERNEL, iterations=1)


def main():
    args = parse_args()

    images = get_image_paths(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for image_file_name in images:
        image_file_path = os.path.join(args.input_dir, image_file_name)
        print(f"INFO: Processing image {image_file_path}")

        pil_image = Image.open(image_file_path).convert("RGB")
        segmentation_classes = discover_segmentation_classes(pil_image)

        # Pad image to allow kernels to operate "offscreen"
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
            if np.sum(boolean_image) / boolean_image.size > BACKGROUND_THRESHOLD:
                print(
                    f"INFO: Class {segmentation_class} detected as background class. Skipping."
                )
                continue

            # Generate scribble
            class_scribble = generate_scribble_for_boolean(boolean_image)

            # Add scribble to output map
            set_colour(
                colour_image=scribble_annotation,
                binary_image=class_scribble.astype(bool),
                colour=np.asarray(segmentation_class[:3], dtype=np.uint8),
            )

        # Save resulting image
        save_image(
            scribble_annotation[
                BORDER_PADDING:-BORDER_PADDING, BORDER_PADDING:-BORDER_PADDING
            ],
            os.path.join(
                args.output_dir,
                f"{os.path.splitext(image_file_name)[0]}_scribble.png",
            ),
        )


main()
