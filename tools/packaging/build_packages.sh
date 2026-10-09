#!/bin/bash
# Builds Ernie's Linux packages (DEB, TGZ, RPM) in manylinux_2_34 (AlmaLinux 9, glibc 2.34), so they
# install on Ubuntu 22.04+, Debian 12+ and RHEL 9+. That is the oldest base Slang's prebuilt
# library runs on: it needs GCC 11's libstdc++ (GLIBCXX_3.4.29).
#
#   docker run --rm -e OWNER=$(id -u):$(id -g) -v "$PWD":/src -w /src \
#       quay.io/pypa/manylinux_2_34_x86_64 tools/packaging/build_packages.sh
#
# Packages land in build/package-deb/packages and build/package-rpm/packages; OWNER, when given,
# gets the build directories back from the container's root.
set -euo pipefail

dnf install -y -q --enablerepo=crb libX11-devel libXrandr-devel libXinerama-devel libXcursor-devel \
    libXi-devel libXext-devel wayland-devel wayland-protocols-devel libxkbcommon-devel rpm-build
command -v ninja >/dev/null || pipx install -q ninja
cmake --version | sed -n 1p
gcc --version | sed -n 1p
ldd --version | sed -n 1p

for kind in deb rpm; do
    cmake --preset package-$kind
    cmake --build --preset package-$kind -j "$(nproc)"
    cpack --preset package-$kind
done

if [ -n "${OWNER:-}" ]; then
    chown -R "$OWNER" build/package-deb build/package-rpm
fi
