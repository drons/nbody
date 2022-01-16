#!/bin/sh

set -eu

export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:./lib
export PATH=/usr/local/cuda/bin:$PATH

nvidia-smi -L

mkdir build-root
cd build-root 
/usr/lib/x86_64-linux-gnu/qt5/bin/qmake ../nbody.pro \
    CONFIG+=release "OPEN_CV_VERSION_MAJOR=4" \
    CONFIG+=NO_UI
make -j2

./test/engine/engine
./test/solver/solver
./test/extrapolator/extrapolator
./test/solvers_equality/solvers_equality
./test/stream/stream
