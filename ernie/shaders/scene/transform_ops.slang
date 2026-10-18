// SPDX-License-Identifier: BSD-3-Clause
/// @file transform_ops.slang
/// @brief A transform program's op kinds and layout, as `TransformGraph::DeviceOp` encodes them; a
/// test holds them to the C++ constants.

/// A folded constant, then each `XformOpKind` after it.
public static const uint kConstant = 0;
public static const uint kTranslate = 1;
public static const uint kScale = 2;
public static const uint kRotateX = 3;
public static const uint kRotateXYZ = 6;
public static const uint kOrient = 12;

/// `DeviceOp::flags`: bit 0 inverts; bits 8 to 15 count the floats the op reads.
public static const uint kInvert = 1;
public static const uint kFloatsShift = 8;

public static const uint kMaxOps = 8;
/// No slot: a root's parent.
public static const uint kNone = 0xffffffff;
