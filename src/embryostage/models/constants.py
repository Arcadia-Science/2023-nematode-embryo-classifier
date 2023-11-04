# the x-y size of the embryo crops
EMBRYO_CROP_SIZE = 224

FEATURES_GROUP_NAME = "dynamic_features"

# the allowed labels for the manual annotations
EMBRYO_STAGE_LABELS = [
    "proliferation",
    "bean",
    "comma",
    "fold",
    "hatch",
    "death",
    "unfertilized",
]

# map between the labels used in the manual annotations and an arbitrary linear index
EMBRYO_STAGE_INDEX_TO_LABEL = {ind: label for ind, label in enumerate(EMBRYO_STAGE_LABELS)}
