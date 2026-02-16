import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE = torch.device("cpu")

ENABLE_QUANTIZATION = os.environ.get("NEURONEST_QUANTIZE", "1") == "1"

FLOOR_CLASSES = {
    "floor": [3, 4, 13],
    "carpet": [28],
    "mat": [78],
}

FLOOR_CLASS_IDS = [3, 4, 13, 28, 78]

# 46 ADE20K classes that are exclusively outdoor (vehicles, terrain, water bodies,
# outdoor structures, vegetation). These are suppressed in visualization and reports
# since NeuroNest analyzes indoor care environments only. The pretrained EoMT model
# still outputs all 150 classes — this is post-processing filtering.
OUTDOOR_CLASS_IDS = frozenset({
    1,   # building
    2,   # sky
    4,   # tree
    9,   # grass
    11,  # sidewalk, pavement
    16,  # mountain, mount
    20,  # car
    21,  # water
    25,  # house
    26,  # sea
    29,  # field
    32,  # fence
    34,  # rock, stone
    46,  # sand
    48,  # skyscraper
    51,  # grandstand, covered stand
    54,  # runway
    60,  # river
    61,  # bridge, span
    68,  # hill
    72,  # palm, palm tree
    76,  # boat
    79,  # hovel, hut, shack
    80,  # bus
    83,  # truck
    84,  # tower
    86,  # awning, sunshade
    87,  # street lamp
    90,  # plane
    91,  # dirt track
    93,  # pole
    102, # van
    103, # ship
    104, # fountain
    106, # canopy
    109, # pool
    111, # barrel, cask
    113, # falls (waterfall)
    114, # tent
    116, # minibike, motorbike
    122, # tank, storage tank
    127, # bicycle
    128, # lake
    136, # traffic light
    140, # pier
    149, # flag
})

# 104 indoor/dual-use classes (complement of OUTDOOR_CLASS_IDS within 0-149)
INDOOR_CLASS_IDS = frozenset(range(150)) - OUTDOOR_CLASS_IDS

BLACKSPOT_MODEL_REPO = "lolout1/txstNeuroNest"
BLACKSPOT_MODEL_FILE = "model_0004999.pth"

DISPLAY_MAX_WIDTH = 1920
DISPLAY_MAX_HEIGHT = 1080

EOMT_MODEL_ID = "tue-mps/ade20k_semantic_eomt_large_512"
