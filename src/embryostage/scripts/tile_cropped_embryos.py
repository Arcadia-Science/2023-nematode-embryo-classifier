import click
import imageio
import numpy as np
import PIL
import zarr

from embryostage.cli import options as cli_options
from embryostage.scripts.batch_classify_embryos import parse_ids_from_embryo_filepath


def tile_embryo(embryo_filepath, subsample_timepoints_by, subsample_xy_by):
    '''
    tile the frames from a cropped embryo timelapse horizontally to form a single 2D image
    of shape (size_x, size_t * size_y)
    '''

    im = zarr.open(embryo_filepath)

    # the raw images are shape (size_t, size_x, size_y)
    im = np.asarray(im).squeeze()

    # subsample to reduce the size of the concatenated image
    im_subsampled = im[::subsample_timepoints_by, ::subsample_xy_by, ::subsample_xy_by]

    size_t, size_x, size_y = im_subsampled.shape

    # zero-pad the image in the x and y directions to create a black border between the frames
    # after they are concatenated
    im_padded = np.zeros((size_t, size_x + 2, size_y + 2))
    im_padded[:, 1:-1, 1:-1] = im_subsampled

    # concat the timepoints in the x direction (by column)
    size_t, size_x, size_y = im_padded.shape
    im_tiled = im_padded.transpose(1, 0, 2).reshape(size_x, size_t * size_y)

    return im_tiled


def normalize_to_uint8(im):
    '''
    normalize an array to 0-1 and convert to uint8
    '''
    im = im.astype(np.float32)
    im -= im.min()
    im /= im.max()
    im *= 255
    im = im.astype(np.uint8)
    return im


def rasterize_text(text, font_size, image_size):
    '''
    rasterize a string to an image and return the image as a numpy array
    '''
    font = PIL.ImageFont.load_default(size=font_size)

    # Create a new blank image ('L' mode for grayscale)
    image = PIL.Image.new('L', image_size, color=255)

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
    '--subsample-embryos-by',
    type=int,
    default=1,
    help='The factor by which to subsample the embryos',
)
@click.option(
    '--subsample-timepoints-by',
    type=int,
    default=10,
    help='The factor by which to subsample timepoints',
)
@click.option(
    '--subsample-xy-by',
    type=int,
    default=2,
    help='The factor by which to subsample in x-y',
)
@click.command()
def main(
    data_dirpath, dataset_id, subsample_embryos_by, subsample_timepoints_by, subsample_xy_by
):
    '''
    tile all cropped embryo timelapses from a given dataset
    by concatenating timelapse frames by column (i.e., horizontally)
    and then concatenating the resulting images by row (i.e., vertically)

    the resulting image is saved to the data directory as a JPEG file
    '''

    # aggregate the filepaths for all embryos from all FOVs in the dataset
    embryo_filepaths = list(
        (data_dirpath / 'cropped_embryos' / dataset_id).glob('fov*/embryo*.zarr')
    )
    if not embryo_filepaths:
        raise FileNotFoundError(
            f"No encoded dynamics data found for dataset '{dataset_id}' in {data_dirpath}"
        )

    # sort the embryo_filepaths by FOV ID
    embryo_filepaths = sorted(
        embryo_filepaths,
        key=lambda filepath: int(parse_ids_from_embryo_filepath(filepath)['fov_id']),
    )

    tiles = []
    for embryo_filepath in embryo_filepaths[::subsample_embryos_by]:
        tile = tile_embryo(
            embryo_filepath,
            subsample_timepoints_by=subsample_timepoints_by,
            subsample_xy_by=subsample_xy_by,
        )
        tile = normalize_to_uint8(tile)

        # create an image containing the embryo and FOV IDs
        ids = parse_ids_from_embryo_filepath(embryo_filepath)
        id_image = rasterize_text(
            text=f"fov{ids['fov_id']}\n{ids['embryo_id']}",
            image_size=(tile.shape[0], tile.shape[0]),
            font_size=10,
        )

        # prepend the image of the IDs to the embryo tile
        tile = np.concatenate((id_image, tile), axis=1)

        tiles.append(tile)

    im = np.concatenate(tuple(tiles), axis=0)
    imageio.imsave(data_dirpath / f'{dataset_id}-cropped-embryos-tile.jpg', im)


if __name__ == '__main__':
    main()
