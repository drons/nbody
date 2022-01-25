#!/bin/sh

set -eu

export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:./lib
export PATH=/usr/local/cuda/bin:$PATH
export NCCL_DEBUG=TRACE
export NCCL_CHECK_DUPLICATE_GPU=0
export NCCL_CHECK_POINTERS=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=lo

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
