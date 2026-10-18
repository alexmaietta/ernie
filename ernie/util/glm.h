// SPDX-License-Identifier: BSD-3-Clause
/// @file glm.h
/// @brief glm as Ernie's public headers use it, with its layout checked.
///
/// Public headers include glm through here. A consumer may build with its own glm, but libernie was
/// built with these layouts: a glm configured otherwise (`GLM_FORCE_DEFAULT_ALIGNED_GENTYPES`, say)
/// fails here instead of corrupting values across the library boundary.
#pragma once

#include <glm/glm.hpp>

static_assert(sizeof(glm::vec2) == 8 && sizeof(glm::vec3) == 12 && sizeof(glm::vec4) == 16,
              "Ernie needs glm's packed float vectors");
static_assert(sizeof(glm::dvec2) == 16 && sizeof(glm::dvec3) == 24 && sizeof(glm::dvec4) == 32,
              "Ernie needs glm's packed double vectors");
static_assert(sizeof(glm::uvec2) == 8 && sizeof(glm::ivec3) == 12,
              "Ernie needs glm's packed integer vectors");
static_assert(sizeof(glm::mat4) == 64 && sizeof(glm::dmat4) == 128 && sizeof(glm::dmat3) == 72,
              "Ernie needs glm's packed matrices");
