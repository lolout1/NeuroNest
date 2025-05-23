"""Global constants and configurations for NeuroNest application."""

# ADE20K class mappings for floor detection
FLOOR_CLASSES = {
    'floor': [3, 4, 13],  # floor, wood floor, rug
    'carpet': [28],       # carpet
    'mat': [78],          # mat
}

# OneFormer configurations
ONEFORMER_CONFIG = {
    "ADE20K": {
        "key": "ade20k",
        "swin_cfg": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
        "swin_model": "shi-labs/oneformer_ade20k_swin_large",
        "swin_file": "250_16_swin_l_oneformer_ade20k_160k.pth",
    }
}

# Simple class names for display (when needed)
ADE20K_CLASS_NAMES = {
    0: 'wall', 1: 'building', 2: 'sky', 3: 'floor', 4: 'tree',
    5: 'ceiling', 6: 'road', 7: 'bed', 8: 'window', 9: 'grass',
    10: 'cabinet', 11: 'sidewalk', 12: 'person', 13: 'earth', 14: 'door',
    15: 'table', 16: 'mountain', 17: 'plant', 18: 'curtain', 19: 'chair',
    20: 'car', 21: 'water', 22: 'painting', 23: 'sofa', 24: 'shelf',
    25: 'house', 26: 'sea', 27: 'mirror', 28: 'rug', 29: 'field',
    30: 'armchair', 31: 'seat', 32: 'fence', 33: 'desk', 34: 'rock',
    35: 'wardrobe', 36: 'lamp', 37: 'bathtub', 38: 'railing', 39: 'cushion',
    40: 'base', 41: 'box', 42: 'column', 43: 'signboard', 44: 'chest of drawers',
    45: 'counter', 46: 'sand', 47: 'sink', 48: 'skyscraper', 49: 'fireplace',
    50: 'refrigerator', 51: 'grandstand', 52: 'path', 53: 'stairs', 54: 'runway',
    55: 'case', 56: 'pool table', 57: 'pillow', 58: 'screen door', 59: 'stairway',
    60: 'river', 61: 'bridge', 62: 'bookcase', 63: 'blind', 64: 'coffee table',
    65: 'toilet', 66: 'flower', 67: 'book', 68: 'hill', 69: 'bench',
    70: 'countertop', 71: 'stove', 72: 'palm', 73: 'kitchen island', 74: 'computer',
    75: 'swivel chair', 76: 'boat', 77: 'bar', 78: 'arcade machine', 79: 'hovel',
    80: 'bus', 81: 'towel', 82: 'light', 83: 'truck', 84: 'tower',
    85: 'chandelier', 86: 'awning', 87: 'streetlight', 88: 'booth', 89: 'television',
    90: 'airplane', 91: 'dirt track', 92: 'apparel', 93: 'pole', 94: 'land',
    95: 'bannister', 96: 'escalator', 97: 'ottoman', 98: 'bottle', 99: 'buffet',
    100: 'poster', 101: 'stage', 102: 'van', 103: 'ship', 104: 'fountain',
    105: 'conveyer belt', 106: 'canopy', 107: 'washer', 108: 'plaything', 109: 'swimming pool',
    110: 'stool', 111: 'barrel', 112: 'basket', 113: 'waterfall', 114: 'tent',
    115: 'bag', 116: 'minibike', 117: 'cradle', 118: 'oven', 119: 'ball',
    120: 'food', 121: 'step', 122: 'tank', 123: 'trade name', 124: 'microwave',
    125: 'pot', 126: 'animal', 127: 'bicycle', 128: 'lake', 129: 'dishwasher',
    130: 'screen', 131: 'blanket', 132: 'sculpture', 133: 'hood', 134: 'sconce',
    135: 'vase', 136: 'traffic light', 137: 'tray', 138: 'ashcan', 139: 'fan',
    140: 'pier', 141: 'crt screen', 142: 'plate', 143: 'monitor', 144: 'bulletin board',
    145: 'shower', 146: 'radiator', 147: 'glass', 148: 'clock', 149: 'flag'
}
