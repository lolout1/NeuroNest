import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install torch first
print("Installing torch...")
install_package("torch==2.0.1+cpu")
install_package("torchvision==0.15.2+cpu")
install_package("numpy==1.24.4")

print("Installing detectron2 dependencies...")
install_package("git+https://github.com/cocodataset/panopticapi.git")
install_package("git+https://github.com/mcordts/cityscapesScripts.git")

print("Installing detectron2...")
install_package("git+https://github.com/facebookresearch/detectron2.git@v0.6")

print("All critical dependencies installed!")
