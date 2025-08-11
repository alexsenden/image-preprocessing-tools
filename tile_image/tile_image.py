import os

from PIL import Image


Image.MAX_IMAGE_PIXELS = None


def split_image_into_tiles(image_path, output_dir, tile_size):
    # Ensure output directory exists
    print(f"Processing image: {image_path}")
    print(f"Saving tiles to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    if os.path.isfile(os.path.join(output_dir, f"{base_name}_tile_0000.png")):
        return  # Skip if tiles already exist

    img = Image.open(image_path)
    img_width, img_height = img.size
    

    tile_count = 0
    y = 0
    while y < img_height:
        # Ensure last row stays within bounds
        if y + tile_size > img_height:
            y = img_height - tile_size

        x = 0
        while x < img_width:
            # Ensure last column stays within bounds
            if x + tile_size > img_width:
                x = img_width - tile_size

            box = (x, y, x + tile_size, y + tile_size)
            tile = img.crop(box)

            tile_filename = f"{base_name}_tile_{tile_count:04d}.png"
            tile.save(os.path.join(output_dir, tile_filename))
            tile_count += 1

            if x + tile_size >= img_width:
                break
            x += tile_size

        if y + tile_size >= img_height:
            break
        y += tile_size

    print(f"Processed {image_path}, created {tile_count} tiles.")


def tile_all_images_in_directory(input_dir, output_dir, tile_size):
    print(f"Tiling all images in directory: {input_dir}")
    for image_file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_file)
        image_name = os.path.splitext(image_file)[0]
        if os.path.isfile(image_path) and image_file.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tif")
        ):
            try:
                split_image_into_tiles(image_path, f"{output_dir}/{image_name}", tile_size)
            except Exception as e:
                print(f"*****************\nError processing {image_file}: {e}\n*****************")
        elif os.path.isdir(image_path):
            tile_all_images_in_directory(
                image_path, f"{output_dir}/{image_name}", tile_size
            )
        else:
            print(f"Skipping non-image file: {image_file}")
