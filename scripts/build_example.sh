#!/bin/bash

if [ $# -ne 1 ]; then
  echo "build_example.sh [example_to_build]" 1>&2
  exit 1
fi

cmake -DOPENCV_DIR=/external/opencv/platforms/linux/build -DTARGET=$1 .
make
mv bin/{example,$1}
