# 2023-celegans-sandbox

[![License MIT](https://img.shields.io/pypi/l/2023-celegans-sandbox.svg?color=green)](https://github.com/Arcadia-Science/2023-celegans-sandbox/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/2023-celegans-sandbox.svg?color=green)](https://pypi.org/project/2023-celegans-sandbox)
[![Python Version](https://img.shields.io/pypi/pyversions/2023-celegans-sandbox.svg?color=green)](https://python.org)
[![tests](https://github.com/Arcadia-Science/2023-celegans-sandbox/workflows/tests/badge.svg)](https://github.com/Arcadia-Science/2023-celegans-sandbox/actions)
[![codecov](https://codecov.io/gh/ArcadiaScience/2023-celegans-sandbox/branch/main/graph/badge.svg)](https://codecov.io/gh/ArcadiaScience/2023-celegans-sandbox)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/2023-celegans-sandbox)](https://napari-hub.org/plugins/2023-celegans-sandbox)

Label-free analysis and prediciton of developmental outcomes of C. elgans embryos
----------------------------------

## Installation

This repository uses conda to manage software environments and installations.
You can find operating system-specific instructions for installing miniconda [here](https://docs.conda.io/en/latest/miniconda.html).
After installing conda and [mamba](https://mamba.readthedocs.io/en/latest/), setup two conda environemnts:
*  `NDutils` to convert image data to ome-zarr format 
*  `celegans` to run this pipeline to predict developmental stages and outcomes. 

### Setup `NDutils` environment for data conversion
Run the following commands in the terminal. 

```sh
mamba create -n NDutils python=3.9
# bioformats2raw conda package installs OpenJDK.
mamba install -c ome bioformats2raw
# pyqt6 package is required on macosx arm64 (with M1 chips).
pip install pyqt6 napari napari-ome-zarr
```

We use zarr format to store the N-dimensional data and metadata. The data acquired from most microscopes can be converted to ome-zarr format using [bioformats2raw] converter.

You may be able to find python libraries for your data formats as well, e.g., [tifffile], [nd2] if you need more fine-grained control over conversion.  


### Setup `celegans` environment for analysis

To setup the conda environment : 
```sh   
    mamba create -n celegans python=3.9
```


To install current development version :
```sh
    pip install git+https://github.com/Arcadia-Science/2023-celegans-sandbox.git
```

This napari plugin is a phython package that can be installed via pip, along with all of its dependencies. 


## Usage

Following are the steps in the analysis:

### convert data to zarr
Activate the conda environment
```sh
conda activate NDutils
```

[Convert the data](https://github.com/glencoesoftware/bioformats2raw)
```sh
bioformats2raw /path/to/file.nd2 /path/to/zarr-pyramid
```

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


Install the editable version of the plugin, along with dependencies needed for development and testing:
```sh
   git clone https://github.com/Arcadia-Science/2023-celegans-sandbox.git
   cd 2023-celegans-sandbox
   pip install -e ."[testing, dev]"
```


Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"2023-celegans-sandbox" is free and open source software

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
