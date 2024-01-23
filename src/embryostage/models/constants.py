# the x-y size of the embryo crops
EMBRYO_CROP_SIZE = 224

FEATURES_GROUP_NAME = "dynamic_features"

# the list of allowed labels for the manual annotations, ordered by developmental progression,
# with 'unfertilized' first and 'death' last
# (the order matters only because of how this list is used in post_process_predictions.py)
EMBRYO_STAGE_LABELS = [
    "unfertilized",
    "proliferation",
    "morphogenesis",
    "fold",
    "hatch",
    "death",
]

# map between the labels used in the manual annotations and an arbitrary linear index
EMBRYO_STAGE_INDEX_TO_LABEL = {ind: label for ind, label in enumerate(EMBRYO_STAGE_LABELS)}

# the random seed to use for reproducibility during training
RANDOM_SEED = 2023
