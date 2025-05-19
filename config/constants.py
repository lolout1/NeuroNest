
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
        "width": 640
    }
}
