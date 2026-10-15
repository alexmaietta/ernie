# SPDX-License-Identifier: BSD-3-Clause
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
COPY ernie_*_amd64.deb /tmp/
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates mesa-vulkan-drivers /tmp/ernie_*_amd64.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/ernie_*_amd64.deb

# The NVIDIA Container Toolkit mounts the host driver's Vulkan ICD with
# these capabilities.
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
WORKDIR /work
ENTRYPOINT ["ernie"]
CMD ["--help"]
