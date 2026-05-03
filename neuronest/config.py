import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE = torch.device("cpu")

ENABLE_QUANTIZATION = os.environ.get("NEURONEST_QUANTIZE", "1") == "1"

FLOOR_CLASSES = {
    "floor": [3],   # 3: floor
    "rug": [28],    # 28: rug
}

FLOOR_CLASS_IDS = [3, 28]

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

# --- Defect 3: Sign & Clock vertical placement ---
# ADE20K class IDs used by the placement analyzer. See ade20k_classes.py for the
# full catalog. Tuples (not sets) so the order is deterministic for logging.
SIGN_CLASS_IDS = (43,)         # 43: signboard, sign
CLOCK_CLASS_IDS = (148,)       # 148: clock
PLACEMENT_TARGET_CLASS_IDS = SIGN_CLASS_IDS + CLOCK_CLASS_IDS

# Calibration reference classes (used by placement/calibration.py to scale depth)
DOOR_CLASS_ID = 14             # 14: door
CEILING_CLASS_ID = 5           # 5: ceiling

# ADA / dementia-design recommended centroid range, in inches above finished floor.
ADA_PLACEMENT_LOW_IN = 48.0
ADA_PLACEMENT_HIGH_IN = 60.0
ADA_PLACEMENT_TOLERANCE_IN = 2.0   # soft band; severity bumped one tier inside

# Severity thresholds (deviation outside the [low, high] band, in inches)
PLACEMENT_SEVERITY_CRITICAL_IN = 6.0
PLACEMENT_SEVERITY_HIGH_IN = 3.0

# Real-world reference dimensions used for self-calibration (inches)
DOOR_REFERENCE_HEIGHT_IN = 80.0     # US standard 6'8" interior door
CEILING_MIN_REFERENCE_IN = 84.0     # plausible indoor ceiling lower bound
CEILING_MAX_REFERENCE_IN = 144.0    # upper bound for sanity check
CALIBRATION_SCALE_MIN = 0.7         # reject scale corrections outside this band
CALIBRATION_SCALE_MAX = 1.4

# Minimum connected-component area (pixels) to count as a real instance
PLACEMENT_MIN_INSTANCE_PIXELS = 100

# Minimum floor pixels required to attempt RANSAC plane fit
PLACEMENT_MIN_FLOOR_PIXELS = 1500

# Monocular metric depth model (Apache 2.0, ~25M params, ~50MB INT8 resident)
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

# Default camera horizontal field-of-view assumption when no EXIF available.
# 60 degrees matches typical phone main-camera FOV (iPhone, modern Android).
DEFAULT_HORIZONTAL_FOV_DEG = 60.0
