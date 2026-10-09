#!/bin/bash
# Installs Ernie's DEBs on a clean Debian or Ubuntu and uses them as an application would: runs the
# viewer's --version, then builds and runs the find_package(Ernie) consumer on lavapipe.
#
#   docker run --rm -v "$PWD":/src:ro -v "$PWD"/build/package-deb/packages:/pkgs:ro ubuntu:22.04 \
#       /src/tools/packaging/test_deb.sh
set -eu
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq /pkgs/*.deb mesa-vulkan-drivers cmake g++ make >/dev/null
dpkg -l | grep ernie | awk '{print $2, $3}'
ernie --version

cmake -S /src/ernie/test/install -B /tmp/consumer -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build /tmp/consumer
VK_DRIVER_FILES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | sed -n 1p)
export VK_DRIVER_FILES
/tmp/consumer/consumer
