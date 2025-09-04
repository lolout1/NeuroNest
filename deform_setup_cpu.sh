#!/usr/bin/env bash

echo '----------------------------------------------------------------'
echo 'CPU Setup - Skipping CUDA operations compilation'
echo '----------------------------------------------------------------'
pip3 freeze | grep torch
pip3 freeze | grep detectron2
pip3 freeze | grep natten
echo '----------------------------------------------------------------'
echo 'CPU setup complete'
echo '----------------------------------------------------------------'
