// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <filesystem>
#include <optional>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "bindings.h"

namespace nb = nanobind;

namespace ernie::python {

namespace {

using HostArray = nb::ndarray<nb::ro, nb::device::cpu, nb::c_contig>;
template <class T> using TypedArray = nb::ndarray<const T, nb::device::cpu, nb::c_contig>;

template <class Array> Pixels ToPixels(const Array& array) {
    if (array.ndim() != 2 && array.ndim() != 3) {
        throw nb::value_error("Images are (rows, cols[, channels]) arrays");
    }
    const auto rows = int64_t(array.shape(0));
    const auto cols = int64_t(array.shape(1));
    const int64_t channels = array.ndim() == 3 ? int64_t(array.shape(2)) : 1;
    const void* data = array.data();
    const nb::dlpack::dtype dtype = array.dtype();
    if (dtype == nb::dtype<double>()) {
        const auto* values = static_cast<const double*>(data);
        std::vector<float> floats(array.size());
        std::transform(values, values + array.size(), floats.begin(),
                       [](double value) { return float(value); });
        return Pixels(std::move(floats), rows, cols, channels);
    }
    const device::DType type =
        dtype == nb::dtype<uint8_t>()    ? device::DType::UInt8
        : dtype == nb::dtype<uint32_t>() ? device::DType::UInt32
        : dtype == nb::dtype<float>()
            ? device::DType::Float32
            : throw nb::type_error("Images are uint8, uint32, float32 or float64 arrays");
    return Pixels(Span<const std::byte>(static_cast<const std::byte*>(data), array.nbytes()), type,
                  rows, cols, channels);
}

nb::object ToNumpy(Pixels pixels) {
    const std::vector<size_t> shape = {size_t(pixels.rows), size_t(pixels.cols),
                                       size_t(pixels.channels)};
    return Numpy(std::move(pixels.bytes), shape, DataType(pixels.type));
}

}

void BindIo(nb::module_& module) {
    module.def(
        "write_image",
        [](const std::filesystem::path& path, const device::Tensor& image) {
            return WriteImage(path, image);
        },
        nb::arg("path"), nb::arg("image"), nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "write_image",
        [](const std::filesystem::path& path, const HostArray& image) {
            return WriteImage(path, ToPixels(image));
        },
        nb::arg("path"), nb::arg("image"));

    nb::enum_<Colormap>(module, "Colormap")
        .value("GRAYSCALE", Colormap::Grayscale)
        .value("TURBO", Colormap::Turbo)
        .value("VIRIDIS", Colormap::Viridis);

    module.def(
        "colorize_scalars",
        [](const device::Tensor& values, Colormap colormap, std::optional<float> minimum,
           std::optional<float> maximum, bool logarithmic, bool reversed, bool centered,
           uint32_t channel) {
            return ColorizeScalars(
                values, {colormap, minimum, maximum, logarithmic, reversed, centered, channel});
        },
        nb::arg("values"), nb::kw_only(), nb::arg("colormap") = Colormap::Grayscale,
        nb::arg("minimum") = nb::none(), nb::arg("maximum") = nb::none(),
        nb::arg("logarithmic") = false, nb::arg("reversed") = false, nb::arg("centered") = false,
        nb::arg("channel") = 0, nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "colorize_scalars",
        [](const TypedArray<float>& values, Colormap colormap, std::optional<float> minimum,
           std::optional<float> maximum, bool logarithmic, bool reversed, bool centered,
           uint32_t channel) {
            return ToNumpy(
                ColorizeScalars(ToPixels(values), {colormap, minimum, maximum, logarithmic,
                                                   reversed, centered, channel}));
        },
        nb::arg("values"), nb::kw_only(), nb::arg("colormap") = Colormap::Grayscale,
        nb::arg("minimum") = nb::none(), nb::arg("maximum") = nb::none(),
        nb::arg("logarithmic") = false, nb::arg("reversed") = false, nb::arg("centered") = false,
        nb::arg("channel") = 0);

    module.def("colorize_hdr", nb::overload_cast<const device::Tensor&, float>(&ColorizeHdr),
               nb::arg("color"), nb::arg("exposure") = 0.0f,
               nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "colorize_hdr",
        [](const TypedArray<float>& color, float exposure) {
            return ToNumpy(ColorizeHdr(ToPixels(color), exposure));
        },
        nb::arg("color"), nb::arg("exposure") = 0.0f);

    module.def("colorize_normals", nb::overload_cast<const device::Tensor&>(&ColorizeNormals),
               nb::arg("normals"), nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "colorize_normals",
        [](const TypedArray<float>& normals) {
            return ToNumpy(ColorizeNormals(ToPixels(normals)));
        },
        nb::arg("normals"));

    module.def("colorize_ids", nb::overload_cast<const device::Tensor&>(&ColorizeIds),
               nb::arg("ids"), nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "colorize_ids",
        [](const TypedArray<uint32_t>& ids) { return ToNumpy(ColorizeIds(ToPixels(ids))); },
        nb::arg("ids"));

    module.def(
        "write_render_var",
        [](const std::filesystem::path& path, const RenderVarOutput& renderVar, int64_t camera) {
            return WriteRenderVar(path, renderVar, camera);
        },
        nb::arg("path"), nb::arg("render_var"), nb::arg("camera") = 0,
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "write_frame",
        [](const std::filesystem::path& directory, const FrameOutput& frame, bool raw) {
            return WriteFrame(directory, frame, raw);
        },
        nb::arg("directory"), nb::arg("frame"), nb::arg("raw") = false,
        nb::call_guard<nb::gil_scoped_release>());

    nb::class_<ImageWriter>(module, "ImageWriter")
        .def(nb::init<uint32_t, uint32_t>(), nb::arg("threads") = 4, nb::arg("max_pending") = 64)
        .def(
            "write_image",
            [](ImageWriter& writer, std::filesystem::path path, const device::Tensor& image) {
                writer.WriteImage(std::move(path), image);
            },
            nb::arg("path"), nb::arg("image"), nb::call_guard<nb::gil_scoped_release>())
        .def(
            "write_image",
            [](ImageWriter& writer, std::filesystem::path path, const HostArray& image) {
                Pixels pixels = ToPixels(image);
                nb::gil_scoped_release release;
                writer.WriteImage(std::move(path), std::move(pixels));
            },
            nb::arg("path"), nb::arg("image"))
        .def("write_render_var", &ImageWriter::WriteRenderVar, nb::arg("path"),
             nb::arg("render_var"), nb::arg("camera") = 0, nb::call_guard<nb::gil_scoped_release>())
        .def("write_frame", &ImageWriter::WriteFrame, nb::arg("directory"), nb::arg("frame"),
             nb::arg("raw") = false, nb::call_guard<nb::gil_scoped_release>())
        .def("wait", &ImageWriter::Wait, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("pending", &ImageWriter::Pending)
        .def(
            "__enter__", [](ImageWriter& writer) -> ImageWriter& { return writer; },
            nb::rv_policy::reference)
        .def(
            "__exit__",
            [](ImageWriter& writer, nb::handle, nb::handle, nb::handle) {
                nb::gil_scoped_release release;
                writer.Wait();
            },
            nb::arg("type").none(), nb::arg("value").none(), nb::arg("traceback").none());
}

}
