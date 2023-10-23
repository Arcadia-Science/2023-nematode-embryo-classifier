# EmbryoStage

Label-free analysis and prediciton of developmental stages of embryos
----------------------------------

## Installation

This repository uses conda to manage software environments and installations.
You can find operating system-specific instructions for installing miniconda [here](https://docs.conda.io/en/latest/miniconda.html).
After installing conda and [mamba](https://mamba.readthedocs.io/en/latest/), setup a new conda environment.

### Setup `embryostage` environment

To setup the conda environment :
```sh
    mamba create -n embryostage python=3.10
```


To install current development version :
```sh
    pip install git+https://github.com/Arcadia-Science/2023-embryostage.git
```



## Usage

Following are the steps in the analysis:

### convert data to zarr
Activate the conda environment
```sh
conda activate embryostage
```


We use zarr format to store the N-dimensional data and metadata. The data acquired from most microscopes can be converted to ome-zarr format using [bioformats2raw] converter.

We use [nd2] and [iohub] to convert the Nikon ND2 format to ome-zarr format.


___________________

See the data and metadata organization [schema](docs/data_schema.md).

### extract movies of embryos

### annotate developmental events

### compute features

### train a classifier

### evaluate a classifier

###  use a classifier

## Contributing


Contributions are very welcome.


## License

Distributed under the terms of the [MIT] license,
"2023-embryostage" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/Arcadia-Science/2023-celegans-sandbox/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[bioformats2raw]: https://github.com/glencoesoftware/bioformats2raw
[tifffile]: https://pypi.org/project/tifffile/
[nd2]: https://pypi.org/project/nd2/
[iohub]: https://github.com/czbiohub-sf/iohub
