// SPDX-License-Identifier: BSD-3-Clause
/// @file raster.h
/// @brief Rasters read a rectangle at a time: TIFF/GeoTIFF, PDS3 and decoded images, mosaics of
/// them, planetary DEMs and the local heightfields drawn of them.
#pragma once

#include "ernie/export.h"

#include "ernie/util/file.h"
#include "ernie/util/tensor.h"

#include "ernie/util/glm.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

// ----------------------------------------------------------------------------------------------
// Rasters

/// @brief How a raster's samples are stored.
enum class GridSampleType {
    UInt8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    Float32,
};

/// @brief The size of a sample type.
/// @param type The type.
/// @return Bytes per sample.
ERNIE_API std::uint32_t ByteSize(GridSampleType type);

/// @brief Decodes one sample in host byte order.
/// @param sample The sample's bytes.
/// @param type   How it is stored.
/// @return Its value.
ERNIE_API double DecodeSample(const std::byte* sample, GridSampleType type);

/// @brief Where an equirectangular raster lies.
struct LatLongGrid {
    double west = -180.0;        ///< Longitude of the west edge, in degrees east.
    double north = 90.0;         ///< Latitude of the north edge, in degrees north.
    double degreesAcross = 0.0;  ///< Degrees per pixel across; positive.
    double degreesDown = 0.0;    ///< Degrees per pixel down; positive.
};

/// @brief A rectangle of a raster's pixels.
struct GridRect {
    std::uint32_t x = 0;       ///< Left column.
    std::uint32_t y = 0;       ///< Top row.
    std::uint32_t width = 0;   ///< Width, in pixels.
    std::uint32_t height = 0;  ///< Height, in pixels.
};

/// @brief A raster image read a rectangle at a time.
/// Lets images larger than memory be baked, resampled and streamed in parts. Read by one thread at
/// a time.
class ERNIE_API GridSource {
  public:
    virtual ~GridSource() = default;

    /// @return Width, in pixels.
    virtual std::uint32_t Width() const = 0;
    /// @return Height, in pixels.
    virtual std::uint32_t Height() const = 0;
    /// @return Samples per pixel (1 grey, 3 RGB, 4 RGBA).
    virtual std::uint32_t Channels() const = 0;
    /// @return How samples are stored.
    virtual GridSampleType Type() const = 0;

    /// @return Bytes per sample.
    std::uint32_t SampleBytes() const {
        return ByteSize(Type());
    }

    /// @brief Reads a rectangle of samples in host byte order.
    /// @param x   The left column.
    /// @param y   The top row.
    /// @param w   The width, in pixels.
    /// @param h   The height, in pixels.
    /// @param out w * h * Channels() * SampleBytes() bytes, rows top to bottom, interleaved.
    /// @throws std::out_of_range If the rectangle passes the image or `out` is too small.
    /// @throws std::runtime_error If the data is damaged.
    virtual void Read(std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h,
                      std::span<std::byte> out) const = 0;

    /// @return The scale in sample * Scale() + Offset(), the physical value (PDS SCALING_FACTOR;
    ///         usually 1).
    virtual double Scale() const {
        return 1.0;
    }
    /// @return The offset to physical values (usually 0).
    virtual double Offset() const {
        return 0.0;
    }
    /// @return The sample marking no data, if any.
    virtual std::optional<double> NoData() const {
        return std::nullopt;
    }
    /// @return Its latitude-longitude placement, if the file says.
    virtual std::optional<LatLongGrid> Grid() const {
        return std::nullopt;
    }
    /// @return Rectangles best read whole one after another, such as a mosaic's tiles; the whole
    ///         image by default.
    virtual std::vector<GridRect> Parts() const {
        return {GridRect{0, 0, Width(), Height()}};
    }
    /// @brief Frees memory held for reading, such as a decoded image; it is reloaded if needed.
    virtual void Release() const {}

    /// @return Every sample, as `Read` of the whole image.
    std::vector<std::byte> ReadAll() const;

  protected:
    /// @brief Checks a `Read`'s rectangle and room.
    /// @throws std::out_of_range If the rectangle passes the image or `out` is too small.
    void CheckRead(std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h,
                   std::span<const std::byte> out) const;
};

/// @brief A grid of like tiles read as one raster.
/// For global mosaics too large for one file, such as Blue Marble's eight 21600 x 21600 tiles.
/// Parts are tiles, decoded one at a time; the grid is the first tile's, spread over the rest.
class ERNIE_API MosaicRaster : public GridSource {
  public:
    /// @brief Opens the tiles.
    /// @param tiles   The tile files, row by row, top to bottom.
    /// @param columns Tiles per row.
    /// @throws std::invalid_argument If the tiles do not fill whole rows or differ; else as
    ///         `OpenRaster`.
    MosaicRaster(const std::vector<std::filesystem::path>& tiles, std::uint32_t columns);

    std::uint32_t Width() const override {
        return columns_ * tileWidth_;
    }
    std::uint32_t Height() const override {
        return rows_ * tileHeight_;
    }
    std::uint32_t Channels() const override {
        return tiles_.front()->Channels();
    }
    GridSampleType Type() const override {
        return tiles_.front()->Type();
    }
    double Scale() const override {
        return tiles_.front()->Scale();
    }
    double Offset() const override {
        return tiles_.front()->Offset();
    }
    std::optional<double> NoData() const override {
        return tiles_.front()->NoData();
    }
    std::optional<LatLongGrid> Grid() const override {
        return tiles_.front()->Grid();
    }
    std::vector<GridRect> Parts() const override;
    void Release() const override;
    void Read(std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h,
              std::span<std::byte> out) const override;

  private:
    std::vector<std::unique_ptr<GridSource>> tiles_;
    std::uint32_t columns_ = 0;
    std::uint32_t rows_ = 0;
    std::uint32_t tileWidth_ = 0;
    std::uint32_t tileHeight_ = 0;
    /// The tile read last, released as another is read.
    mutable std::size_t current_ = 0;
};

/// @brief Opens the mosaic a `.mosaic` file describes.
/// Each line is a row of tile paths relative to the file, top to bottom, separated by spaces; `#`
/// starts a comment.
/// @param path The `.mosaic` file.
/// @return The mosaic.
/// @throws std::runtime_error If it cannot be read.
/// @throws std::invalid_argument For rows of different lengths, or as `MosaicRaster`.
ERNIE_API std::unique_ptr<MosaicRaster> OpenMosaic(const std::filesystem::path& path);

/// @brief Opens a raster by extension.
/// TIFF/GeoTIFF and PDS3 (.img with .lbl, or the .lbl) read in place; 8/16-bit PNG and JPEG decode
/// whole on first read; `.mosaic` as `OpenMosaic`.
/// @param path The file.
/// @return The raster.
/// @throws std::runtime_error If it cannot be read.
ERNIE_API std::unique_ptr<GridSource> OpenRaster(const std::filesystem::path& path);

// ----------------------------------------------------------------------------------------------
// Grid views

/// @brief A non-owning, north-up grid of `float` samples.
/// Ernie takes heightfields and albedo maps in this form. Row 0 is the northern edge; the world
/// frame is x east, y north, z up, in camera units.
struct GridView {
    const float* data = nullptr;  ///< The first sample.
    std::size_t width = 0;        ///< Columns.
    std::size_t height = 0;       ///< Rows.
    std::size_t rowStride = 0;    ///< Elements from one row to the next; 0 means `width`.
    double originX = 0.0;         ///< West edge of the first cell, in world units.
    double originY = 0.0;         ///< North edge of the first cell, in world units.
    double cellSize = 1.0;        ///< Side of a cell in world units; positive.
    std::optional<float> noData;  ///< Value marking cells without data, if any.

    /// @return The row stride in elements.
    std::size_t Stride() const {
        return rowStride == 0 ? width : rowStride;
    }
    /// @brief Reads one cell.
    /// @param col The column.
    /// @param row The row, 0 at the north.
    /// @return The sample.
    float operator()(std::size_t col, std::size_t row) const {
        return data[row * Stride() + col];
    }
    /// @brief Tests a sample for data.
    /// @param value The sample.
    /// @return Whether it is finite and not the nodata value.
    bool IsValid(float value) const {
        return std::isfinite(value) && !(noData && value == *noData);
    }
};

/// @brief Checks that a grid view is usable.
/// @param view    The view to check.
/// @param what    Its name, for the error message.
/// @param minSide The fewest cells allowed along each side.
/// @throws std::invalid_argument If it has no data, too few cells, a row stride below its width, a
///         non-finite origin, or a cell size that is not positive and finite.
ERNIE_API void CheckGridView(const GridView& view, const std::string& what, std::size_t minSide);

// ----------------------------------------------------------------------------------------------
// PDS3

/// @brief A PDS3 label's keywords, in the order the label gives them.
struct ERNIE_API Pds3Label {
    /// @brief One `KEY = value` line.
    struct Keyword {
        /// The OBJECTs holding it, outermost first, each followed by a dot (`IMAGE.`); empty at the
        /// top level.
        std::string object;
        /// Its key, as the label writes it.
        std::string key;
        /// Its value as written: comments dropped and continued lines joined, units (`<km>`) and
        /// quotes kept.
        std::string value;
    };

    /// The keywords; OBJECT and END_OBJECT lines are not among them.
    std::vector<Keyword> keywords;

    /// @brief A keyword by its object path and key.
    /// @param name Its objects and key, such as `IMAGE.LINES`; case matters.
    /// @return Its value without units or quotes, the label's last of that name, or none.
    std::optional<std::string> Get(std::string_view name) const;
    /// @brief A keyword by its object path and key, as a number.
    /// @param name Its objects and key, such as `IMAGE.LINES`.
    /// @return Its value.
    /// @throws std::invalid_argument If it is missing or not a number.
    double Number(std::string_view name) const;
    /// @brief A keyword by its object path and key.
    /// @param name Its objects and key, such as `IMAGE.LINES`.
    /// @return Its value without units or quotes.
    /// @throws std::invalid_argument If it is missing.
    std::string Text(std::string_view name) const;
    /// @brief A keyword by its key alone, in any object.
    /// @param key Its key, in any case.
    /// @return Its value as written, the label's first with that key, or none.
    std::optional<std::string> Find(std::string_view key) const;
};

/// @brief A PDS3 value's text without its units and outer quotes, as `Pds3Label::Get` gives it.
/// @param value The value as written; a sequence keeps its units.
/// @return Its text.
ERNIE_API std::string CleanPds3Value(std::string_view value);

/// @brief Parses a PDS3 label's text up to its END line.
/// @param text The label: `KEY = value` lines, OBJECT and END_OBJECT nesting, comments (`/* */`)
///             and values continued over lines inside quotes, parentheses or braces.
/// @return Its keywords.
///
/// References:
/// - NASA PDS, "Planetary Data System Standards Reference" 3.8, 2009 (the label's syntax,
///   chapter 12).
ERNIE_API Pds3Label ParsePds3Label(std::string_view text);

/// @brief A PDS3 image read in place from its mapped file.
/// Supports one band of 8/16/32-bit integers or floats in either byte order, SCALING_FACTOR,
/// OFFSET, MISSING_CONSTANT and simple cylindrical grids. The label is attached or in a .lbl
/// beside it.
///
/// References:
/// - NASA PDS, "Planetary Data System Standards Reference" 3.8, 2009 (the label's syntax, chapter
///   12; IMAGE objects, appendix A.21; IMAGE_MAP_PROJECTION, appendix A.22).
/// - LOLA team, "LRO LOLA Gridded Data Record" (LDEM_*.LBL).
class ERNIE_API PdsImage : public GridSource {
  public:
    /// @brief Opens an image.
    /// @param path The label or data file; the label's ^IMAGE names the data beside it.
    /// @throws std::runtime_error If the label cannot be found or read, or describes unsupported
    ///         samples.
    explicit PdsImage(const std::filesystem::path& path);

    std::uint32_t Width() const override {
        return width_;
    }
    std::uint32_t Height() const override {
        return height_;
    }
    std::uint32_t Channels() const override {
        return 1;
    }
    GridSampleType Type() const override {
        return type_;
    }
    double Scale() const override {
        return scale_;
    }
    double Offset() const override {
        return offset_;
    }
    std::optional<double> NoData() const override {
        return noData_;
    }
    std::optional<LatLongGrid> Grid() const override {
        return grid_;
    }
    void Read(std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h,
              std::span<std::byte> out) const override;

    /// @brief Reads a label value, without quotes or units.
    /// @param key The key; the first object naming it wins.
    /// @return The value, or none if absent.
    std::optional<std::string> Label(std::string_view key) const;

    /// @return The whole label, for keywords by their object path.
    const Pds3Label& Keywords() const {
        return label_;
    }

  private:
    Pds3Label label_;
    std::shared_ptr<const MappedFile> file_;
    std::uint64_t start_ = 0;  ///< The first sample's byte offset.
    bool bigEndian_ = false;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    GridSampleType type_ = GridSampleType::UInt8;
    double scale_ = 1.0;
    double offset_ = 0.0;
    std::optional<double> noData_;
    std::optional<LatLongGrid> grid_;
};

// ----------------------------------------------------------------------------------------------
// TIFF

/// @brief A TIFF or GeoTIFF image decoded on demand from its mapped file, so images larger than
/// memory read in parts.
/// Classic and BigTIFF in either byte order; strips or tiles; 8/16/32-bit integers and 32-bit
/// floats; none, LZW or Deflate with predictors 2 and 3. GeoTIFF pixel scale, tie point and GDAL
/// no-data.
///
/// References:
/// - Adobe, "TIFF Revision 6.0", 1992 (the layout, LZW, predictor 2).
/// - Adobe, "Adobe Photoshop TIFF Technical Note 3", 2005 (Deflate, the floating-point predictor).
/// - Aware Systems, "BigTIFF" (64-bit offsets).
/// https://www.awaresystems.be/imaging/tiff/bigtiff.html
/// - OGC, "GeoTIFF Standard" 1.1, 2019 (ModelPixelScale, ModelTiepoint); GDAL's GDAL_NODATA tag
///   (42113).
class ERNIE_API TiffImage : public GridSource {
  public:
    /// @brief Opens an image and reads its layout.
    /// @param path The file.
    /// @throws std::runtime_error If the file cannot be read, is not a TIFF, or has an unsupported
    ///         layout or encoding.
    explicit TiffImage(const std::filesystem::path& path);

    std::uint32_t Width() const override {
        return width_;
    }
    std::uint32_t Height() const override {
        return height_;
    }
    std::uint32_t Channels() const override {
        return channels_;
    }
    GridSampleType Type() const override {
        return type_;
    }
    /// @return GeoTIFF's ModelPixelScale (x, y, z), if present.
    std::optional<std::array<double, 3>> PixelScale() const {
        return pixelScale_;
    }
    /// @return GeoTIFF's ModelTiepoint (raster i, j, k to model x, y, z), if present.
    std::optional<std::array<double, 6>> Tiepoint() const {
        return tiepoint_;
    }
    /// @return GDAL's no-data value, if present.
    std::optional<double> NoData() const override {
        return noData_;
    }
    /// @return The grid from the tie point and pixel scale, when both exist and the model space is
    ///         in degrees.
    std::optional<LatLongGrid> Grid() const override;
    void Read(std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h,
              std::span<std::byte> out) const override;

  private:
    /// Decodes a strip or tile to host order, predictor undone. The last one decoded per plane is
    /// kept for the next `Read`.
    const std::vector<std::byte>& Chunk(std::size_t chunk) const;

    /// Samples per pixel in a chunk: all channels, or one when planes are stored apart.
    std::uint32_t ChunkChannels() const {
        return planes_ > 1 ? 1u : channels_;
    }

    std::shared_ptr<const MappedFile> file_;
    bool bigEndian_ = false;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint32_t channels_ = 1;
    std::uint32_t bits_ = 8;
    GridSampleType type_ = GridSampleType::UInt8;
    std::uint32_t compression_ = 1;
    std::uint32_t predictor_ = 1;
    /// Strips are `width_` wide and `chunkHeight_` rows; tiles `chunkWidth_` by `chunkHeight_`,
    /// `chunksAcross_` to a row.
    bool tiled_ = false;
    std::uint32_t chunkWidth_ = 0;
    std::uint32_t chunkHeight_ = 0;
    std::uint32_t chunksAcross_ = 1;
    /// Chunks of one plane; with the planes stored apart (PlanarConfiguration 2), `planes_` of
    /// them one after another.
    std::size_t chunksPerPlane_ = 0;
    std::uint32_t planes_ = 1;
    std::vector<std::uint64_t> offsets_;
    std::vector<std::uint64_t> byteCounts_;
    std::optional<std::array<double, 3>> pixelScale_;
    std::optional<std::array<double, 6>> tiepoint_;
    std::optional<double> noData_;
    mutable std::vector<std::size_t> cachedChunks_;
    mutable std::vector<std::vector<std::byte>> cached_;
};

// ----------------------------------------------------------------------------------------------
// Planetary DEMs
//
// Digital elevation models as the PDS archives them, and the local heightfields Ernie's Terrain
// draws of them.
//
// References:
// - G. A. Neumann, "Lunar Reconnaissance Orbiter Lunar Orbiter Laser Altimeter Reduced Data Record
//   and Derived Products Software Interface Specification", version 2.42, 2010 (the LDEM products,
//   their labels and simple cylindrical grids).
// - J. P. Snyder, "Map Projections: A Working Manual", USGS Professional Paper 1395, 1987 (the
//   equirectangular projection; the polar stereographic, eq. 21-33 and 21-34).

/// @brief A planet's radius on an equirectangular grid, as LOLA's LDEM products (simple
/// cylindrical, true at the equator) and LROC's NAC DTMs (true at a standard parallel) hold it.
/// `heights` are metres above `referenceRadius` (NaN where the product has no data), row by row
/// from the northern, each row from the western sample eastwards, `pixelsPerDegree` along the
/// meridians and that times cos(standardParallel) along the parallels. Sample s of line l is
/// centred at latitude maxLatitude - (l + 0.5) / ppd and east longitude
/// westLongitude + (s + 0.5) / (ppd cos(standardParallel)).
struct ERNIE_API CylindricalDem {
    std::uint32_t lines = 0;
    std::uint32_t samples = 0;
    double pixelsPerDegree = 1.0;
    double maxLatitudeDeg = 90.0;
    double westLongitudeDeg = 0.0;
    double standardParallelDeg = 0.0;
    double referenceRadius = 0.0;  ///< Metres.
    std::vector<float> heights;

    /// @brief The height at a point, bilinear between the cells' centres, clamped to the grid's
    /// edges.
    /// @param latitudeDeg  The latitude.
    /// @param longitudeDeg The east longitude, any multiple of 360 apart.
    /// @return The height, or NaN beside a cell without data.
    double HeightAt(double latitudeDeg, double longitudeDeg) const;
};

/// @brief Reads the DEM a PDS3 label describes, on its simple cylindrical or equirectangular grid
/// (true at CENTER_LATITUDE).
/// Heights are samples times SCALING_FACTOR, plus OFFSET unless OFFSET is the reference sphere's
/// radius (as LOLA's are); PDS's real null (below -1e30), MISSING_CONSTANT and non-finite samples
/// are no data.
/// @param labelPath The label (or the data, with its label attached or beside it).
/// @return The DEM.
/// @throws std::runtime_error If a file cannot be read.
/// @throws std::invalid_argument For another projection or a label it cannot follow.
ERNIE_API CylindricalDem ReadCylindricalDem(const std::filesystem::path& labelPath);

/// @brief A planet's radius on a polar stereographic grid (spherical, true at the pole), as
/// LOLA's polar LDEM products hold it.
/// `heights` are metres above `referenceRadius`, row by row, `mapScale` metres a pixel at the
/// pole. A point at polar distance rho = 2 R tan(45 + lat / 2) (south) or 2 R tan(45 - lat / 2)
/// (north) and longitude lon from the centre longitude lies at x = rho sin(lon) and y =
/// rho cos(lon) (south) or -rho cos(lon) (north), the cell centre at line
/// lineProjectionOffset - y / mapScale and sample sampleProjectionOffset + x / mapScale.
struct ERNIE_API PolarStereographicDem {
    std::uint32_t lines = 0;
    std::uint32_t samples = 0;
    double mapScale = 1.0;  ///< Metres a pixel.
    double lineProjectionOffset = 0.0;
    double sampleProjectionOffset = 0.0;
    double centerLatitudeDeg = -90.0;  ///< -90 or 90.
    double centerLongitudeDeg = 0.0;
    double referenceRadius = 0.0;  ///< Metres.
    std::vector<float> heights;

    /// @brief The fractional (line, sample) of a point.
    /// @param latitudeDeg  The latitude.
    /// @param longitudeDeg The east longitude.
    /// @return Its line and sample.
    glm::dvec2 ToGrid(double latitudeDeg, double longitudeDeg) const;
    /// @brief The height at a point, bilinear between the cells' centres, clamped to the grid's
    /// edges.
    /// @param latitudeDeg  The latitude.
    /// @param longitudeDeg The east longitude.
    /// @return The height.
    double HeightAt(double latitudeDeg, double longitudeDeg) const;
};

/// @brief Reads a DEM as `ReadCylindricalDem` does, but on a polar stereographic grid.
/// @param labelPath The label.
/// @return The DEM.
/// @throws std::invalid_argument As `ReadCylindricalDem`, and for a projection not centred on a
///         pole.
ERNIE_API PolarStereographicDem ReadPolarStereographicDem(const std::filesystem::path& labelPath);

/// @brief A heightfield of a DEM about one point, in a local frame there: x east, y north and z
/// up, its origin on the reference sphere at the point.
/// Each cell holds the height above that tangent plane of the surface straight below or above it,
/// so the planet's curvature is in the heights (a drop of about x^2 / 2R). Where the DEM has no
/// data the surface is the reference sphere.
struct LocalTerrain {
    Tensor heights;     ///< float32 (rows, cols), the first northern.
    glm::dvec2 origin;  ///< x and y of the north-western cell.
    double cellSize = 1.0;
    /// Body-fixed coordinates (metres, the planet's centre at the origin) to the local frame.
    glm::dmat4 bodyToLocal{1.0};
};

/// @brief Makes the heightfield of a DEM about a point.
/// @param dem          The DEM.
/// @param latitudeDeg  The point's latitude.
/// @param longitudeDeg The point's east longitude.
/// @param cells        Cells along each side, at least 2.
/// @param cellSize     Metres between cells; positive.
/// @return The `cells` x `cells` heightfield.
/// @throws std::invalid_argument For fewer than 2 cells or a non-positive cell size.
ERNIE_API LocalTerrain MakeLocalTerrain(const CylindricalDem& dem, double latitudeDeg,
                                        double longitudeDeg, std::uint32_t cells, double cellSize);
/// @copydoc MakeLocalTerrain(const CylindricalDem&, double, double, std::uint32_t, double)
ERNIE_API LocalTerrain MakeLocalTerrain(const PolarStereographicDem& dem, double latitudeDeg,
                                        double longitudeDeg, std::uint32_t cells, double cellSize);

/// @brief Makes a heightfield as `MakeLocalTerrain` does, taking `fallback`'s heights wherever
/// `dem` has none: a high-resolution DTM inside a regional DEM. Both share `dem`'s reference
/// radius.
/// @param dem          The DEM.
/// @param fallback     The DEM used where `dem` has no data.
/// @param latitudeDeg  The point's latitude.
/// @param longitudeDeg The point's east longitude.
/// @param cells        Cells along each side, at least 2.
/// @param cellSize     Metres between cells; positive.
/// @return The heightfield.
ERNIE_API LocalTerrain MakeLocalTerrain(const CylindricalDem& dem, const CylindricalDem& fallback,
                                        double latitudeDeg, double longitudeDeg,
                                        std::uint32_t cells, double cellSize);

/// @brief The body-fixed unit vector of a planetocentric latitude and east longitude.
/// @param latitudeDeg  The latitude.
/// @param longitudeDeg The east longitude.
/// @return The direction.
ERNIE_API glm::dvec3 BodyDirection(double latitudeDeg, double longitudeDeg);

}  // namespace ernie
