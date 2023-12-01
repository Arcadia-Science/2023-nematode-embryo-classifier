import click
import imageio
import numpy as np
import PIL
import zarr

from embryostage import cli_options
from embryostage.scripts.batch_classify_embryos import parse_ids_from_embryo_filepath


def _tile_embryo(embryo_filepath, subsample_timepoints_by, subsample_xy_by):
    """
    tile the frames from a cropped embryo timelapse horizontally to form a single 2D image
    of shape (size_x, size_t * size_y)
    """

    im = zarr.open(embryo_filepath)

    # the raw images are shape (size_t, size_x, size_y)
    im = np.asarray(im).squeeze()

    # subsample to reduce the size of the concatenated image
    im_subsampled = im[::subsample_timepoints_by, ::subsample_xy_by, ::subsample_xy_by]

    # zero-pad the image in the x and y directions to create a black border between the frames
    # after they are concatenated
    pad_width = 1
    size_t, size_x, size_y = im_subsampled.shape
    im_padded = np.zeros(
        (size_t, size_x + 2 * pad_width, size_y + 2 * pad_width), dtype=im.dtype
    )
    im_padded[:, pad_width:-pad_width, pad_width:-pad_width] = im_subsampled

    # concat the timepoints in the x direction (by column)
    size_t, size_x, size_y = im_padded.shape
    im_tiled = im_padded.transpose(1, 0, 2).reshape(size_x, size_t * size_y)

    return im_tiled


def _normalize_to_uint8(array, percentile=99):
    """
    normalize an array to 0-1 and convert to uint8

    array: the array to normalize
    percentile: the percentile to use for the max and min values
    """
    array = array.astype(np.float32)

    min_val, max_val = np.percentile(array.flatten(), (100 - percentile, percentile))

    array -= min_val
    array /= max_val - min_val

    array[array < 0] = 0
    array[array > 1] = 1

    array *= 255
    array = array.astype(np.uint8)
    return array


def _rasterize_text(text, font_size, image_size):
    """
    rasterize a string to an image and return the image as a numpy array
    """
    font = PIL.ImageFont.load_default(size=font_size)

    # Create a new blank image ('L' mode for grayscale)
    image = PIL.Image.new("L", image_size, color=255)

    # draw the text onto the image, roughly centered vertically
    text_x = 10
    text_y = image_size[1] // 2 - font_size // 2
    draw = PIL.ImageDraw.Draw(image)
    draw.text((text_x, text_y), text, font=font, fill=0)

    image_array = np.array(image)
    return image_array


@cli_options.data_dirpath_option
@cli_options.dataset_id_option
@click.option(
    "--subsample-embryos-by",
    type=int,
    default=1,
    help="The factor by which to subsample the embryos",
)
@click.option(
    "--subsample-timepoints-by",
    type=int,
    default=10,
    help="The factor by which to subsample timepoints",
)
@click.option(
    "--subsample-xy-by",
    type=int,
    default=2,
    help="The factor by which to subsample in x-y",
)
@click.command()
def main(
    data_dirpath, dataset_id, subsample_embryos_by, subsample_timepoints_by, subsample_xy_by
):
    """
    tile all cropped embryo timelapses from a given dataset
    by concatenating timelapse frames by column (i.e., horizontally)
    and concatenating the resulting images by row (i.e., vertically),
    then write the resulting image to the data directory as a JPEG.

    Parameters
    ----------
    data_dirpath: str
        The path to the data directory containing the cropped_embryo directory
    dataset_id: str
        The ID of the dataset to tile
    subsample_embryos_by: int
        The factor by which to subsample the embryos; for example, if a value of 2 is given,
        then every second embryo will be included in the tiled image
    subsample_timepoints_by: int
        The factor by which to subsample timepoints; analogous to subsample_embryos_by,
        but applied to the frames of each timelapse
    subsample_xy_by: int
        The factor by which to subsample the pixels in each timelapse frame;
        in other words, the factor by which to downscale the image in the x and y directions
    """

    # aggregate the filepaths for all embryos from all FOVs in the dataset
    embryo_filepaths = list(
        (data_dirpath / "cropped_embryos" / dataset_id).glob("fov*/embryo*.zarr")
    )
    if not embryo_filepaths:
        raise FileNotFoundError(
            f"No encoded dynamics data found for dataset '{dataset_id}' in {data_dirpath}"
        )

    # sort the embryo_filepaths by fov_id
    embryo_filepaths = sorted(
        embryo_filepaths,
        key=lambda filepath: int(parse_ids_from_embryo_filepath(filepath)["fov_id"]),
    )

    tiled_embryos = []
    for embryo_filepath in embryo_filepaths[::subsample_embryos_by]:
        tiled_embryo = _tile_embryo(
            embryo_filepath,
            subsample_timepoints_by=subsample_timepoints_by,
            subsample_xy_by=subsample_xy_by,
        )
        tiled_embryo = _normalize_to_uint8(tiled_embryo)

        # create an image containing the embryo ID and FOV ID
        # and prepend it to the tiled image as a crude way of labeling the embryos
        # note: this is a *very* crude way to label the embryos in the tiled image;
        # it is intended only as a kind of internal control and sanity check,
        # and not for public consumption
        ids = parse_ids_from_embryo_filepath(embryo_filepath)
        id_image = _rasterize_text(
            text=f"fov{ids['fov_id']}\n{ids['embryo_id']}",
            image_size=(tiled_embryo.shape[0], tiled_embryo.shape[0]),
            font_size=10,
        )
        tiled_embryo = np.concatenate((id_image, tiled_embryo), axis=1)

        tiled_embryos.append(tiled_embryo)

    tiled_array = np.concatenate(tuple(tiled_embryos), axis=0)

    filepath = data_dirpath / f"{dataset_id}-cropped-embryos-tile.jpg"
    imageio.imsave(filepath, tiled_array)
    print(f"Tiled array saved to {filepath}")


if __name__ == "__main__":
    main()
