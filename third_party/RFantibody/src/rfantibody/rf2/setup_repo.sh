#!/bin/bash

git submodule update --init --recursive

# Navigate to the submodule directory and run make
cd submodules/USalign/ && make && cd ../..
