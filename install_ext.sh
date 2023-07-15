#!/bin/bash

ext_path=$1
cd ${ext_path}
rm -rf build
find . -name "*.so" | xargs rm
pip install -v -e .