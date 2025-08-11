# Image Preprocessing Tools

An assortment of command line tools that I occasionally have a need for when processing image datasets, but are too small, niche, or too commonly implemented to have much standalone appeal.

### Included Tools:

#### copy_if_caption_exists

Copies images to the caption directory if there is a caption for the image. Useful for forming datasets from incomplete caption sets.

```
python copy_if_caption_exists.py --image_dir=<image_dir> --caption_dir=<caption_dir>
```

#### tile_image

Basic tool for tiling large image datasets.

```
python tile_image_cli.py --input_dir=<input_dir> --output_dir=<output_dir> [--tile_size=<tile_size_px>]
```
