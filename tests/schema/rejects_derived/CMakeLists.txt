# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.25)
project(ErnieSchemaRejectsDerived NONE)
include("${ERNIE_ROOT}/cmake/ErnieSchema.cmake")
file(
    WRITE "${CMAKE_BINARY_DIR}/bad.schema"
    "class BadThing : Typed (\n    doc = \"A prim with a derived attribute.\"\n) {\n"
    "    float ernie:computed = 0 (\n        derived\n        doc = \"Computed.\"\n    )\n}\n"
)
ernie_generate_schemas(
    "${CMAKE_BINARY_DIR}/generated"
    "${ERNIE_ROOT}/ernie/schema/geom.schema"
    "${ERNIE_ROOT}/ernie/schema/lux.schema"
    "${ERNIE_ROOT}/ernie/schema/sensor.schema"
    "${ERNIE_ROOT}/ernie/schema/render.schema"
    "${ERNIE_ROOT}/ernie/schema/shade.schema"
    "${CMAKE_BINARY_DIR}/bad.schema"
)
