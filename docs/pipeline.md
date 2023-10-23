# EmryoStage pipeline
# convert nd2 files to ome-zarr store
* Use [nd2omezarr](embryostage/scripts/nd2omezarr.py) 
* After the conversion you should have a zarr store with this structure.
```
DQM327_heatshock_230817_raw.zarr/
├── fov0
│   └── raw
├── fov1
│   └── raw
```
* FOVs can be viewed with napari

```
napari --plugin napari-ome-zarr DQM327_heatshock_230817_raw.zarr/fov0/ DQM327_heatshock_230817_raw.zarr/fov10 &
```
![raw images](fovs_raw.png)

## extract movies of embryos

## annotate developmental events

## compute features

## train a classifier

## evaluate a classifier

##  use a classifier