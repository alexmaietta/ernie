# SPDX-License-Identifier: BSD-3-Clause
FROM ubuntu:24.04 AS build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cmake g++ git ninja-build python3-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/ernie \
    && cmake --build /build \
    && cmake --install /build --component app \
    && cmake --install /build --component runtime

FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/ernie/bin/ernie /usr/local/bin/ernie
# Its private libraries, as Slang's compiler, where its rpath ($ORIGIN/../lib/ernie) looks.
COPY --from=build /opt/ernie/lib/ernie /usr/local/lib/ernie
# Data and the licences of everything the program carries.
COPY --from=build /opt/ernie/share /usr/local/share
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
WORKDIR /work
ENTRYPOINT ["ernie"]
CMD ["--help"]
