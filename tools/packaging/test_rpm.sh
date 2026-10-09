#!/bin/bash
# Installs Ernie's RPMs on a clean RHEL-family system and uses them as an application would: runs
# the viewer's --version, then builds and runs the find_package(Ernie) consumer on lavapipe.
#
#   docker run --rm -v "$PWD":/src:ro -v "$PWD"/build/package-rpm/packages:/pkgs:ro rockylinux:9 \
#       /src/tools/packaging/test_rpm.sh
set -eu

dnf install -y -q /pkgs/*.rpm mesa-vulkan-drivers cmake gcc-c++ make >/dev/null
rpm -qa | grep ernie
ernie --version

cmake -S /src/ernie/test/install -B /tmp/consumer -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build /tmp/consumer
VK_DRIVER_FILES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | sed -n 1p)
export VK_DRIVER_FILES
/tmp/consumer/consumer
