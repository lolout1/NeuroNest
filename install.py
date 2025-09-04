import subprocess
import sys
import os

# Install torch first
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.0.0", "torchvision>=0.15.0"])

# Clone and install detectron2
if not os.path.exists('detectron2'):
    subprocess.check_call(["git", "clone", "https://github.com/facebookresearch/detectron2"])

# Install detectron2 dependencies
import distutils.core
dist = distutils.core.run_setup("./detectron2/setup.py")
deps = ' '.join([f"'{x}'" for x in dist.install_requires if 'torch' not in x])
subprocess.check_call(f"{sys.executable} -m pip install {deps}", shell=True)

# Add detectron2 to path
sys.path.insert(0, os.path.abspath('./detectron2'))
