// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/library.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "support/support.h"

namespace ernie {
namespace {

#if defined(_WIN32)
constexpr const char* kLoaded = "kernel32.dll";
constexpr const char* kLoadedSymbol = "GetTickCount";
constexpr const char* kUnloaded = "winmm.dll";
constexpr const char* kOtherUnloaded = "version.dll";
#elif defined(__APPLE__)
constexpr const char* kLoaded = "libSystem.B.dylib";
constexpr const char* kLoadedSymbol = "malloc";
constexpr const char* kUnloaded = "libz.1.dylib";
constexpr const char* kOtherUnloaded = "libexpat.1.dylib";
#else
constexpr const char* kLoaded = "libc.so.6";
constexpr const char* kLoadedSymbol = "malloc";
constexpr const char* kUnloaded = "libresolv.so.2";
constexpr const char* kOtherUnloaded = "libutil.so.1";
#endif

constexpr const char* kMissing = "libernie-no-such-library";

TEST(LibraryTest, LoadPrefersALibraryAlreadyInTheProcess) {
    const test::CapturedLog log;
    const Library library = Library::Load({kMissing, kLoaded});
    ASSERT_TRUE(library);
    EXPECT_EQ(library.Name(), kLoaded);
    ASSERT_EQ(log.Messages().size(), 1u);
    EXPECT_EQ(log.Messages()[0].second,
              std::string("using the ") + kLoaded + " already in the process");
}

TEST(LibraryTest, LoadOpensTheFirstNameThatExists) {
    const test::CapturedLog log;
    const Library library = Library::Load({kMissing, kUnloaded});
    ASSERT_TRUE(library);
    EXPECT_EQ(library.Name(), kUnloaded);
    ASSERT_EQ(log.Messages().size(), 1u);
    EXPECT_EQ(log.Messages()[0].second, std::string("loaded ") + kUnloaded);
}

TEST(LibraryTest, LoadFallsBackToPaths) {
    const Library library = Library::Load({kMissing}, {kMissing, kOtherUnloaded});
    ASSERT_TRUE(library);
    EXPECT_EQ(library.Name(), kOtherUnloaded);
}

#ifdef ERNIE_TEST_BESIDE_DIR
TEST(LibraryTest, LoadLooksBesideADirectoryBeforeTheSearchPath) {
    const std::filesystem::path directory = ERNIE_TEST_BESIDE_DIR;
    EXPECT_FALSE(Library::Load({"libernie_beside.so.1"}));

    const Library beside = Library::Load({"libernie_beside.so.1"}, {}, directory);
    ASSERT_TRUE(beside);
    EXPECT_EQ(beside.Name(), (directory / "libernie_beside.so").string());
    EXPECT_EQ(std::filesystem::canonical(beside.Directory()),
              std::filesystem::canonical(directory));
    int (*answer)() = nullptr;
    ASSERT_TRUE(beside.Symbol("ernie_beside", answer));
    EXPECT_EQ(answer(), 7);

    const test::CapturedLog log;
    const Library loaded = Library::Load({"libernie_beside.so.1"});
    ASSERT_TRUE(loaded);
    EXPECT_EQ(loaded.Name(), "libernie_beside.so");
    ASSERT_EQ(log.Messages().size(), 1u);
    EXPECT_EQ(log.Messages()[0].second, "using the libernie_beside.so already in the process");
}
#endif

TEST(LibraryTest, ReportsTheDirectoryItLoadedFrom) {
    const Library library = Library::Load({kLoaded});
#if defined(__APPLE__)
    EXPECT_TRUE(library.Directory().empty());
#else
    EXPECT_TRUE(std::filesystem::exists(library.Directory()));
#endif
    EXPECT_TRUE(Library().Directory().empty());
}

TEST(LibraryTest, IsEmptyWhenNothingLoads) {
    const test::CapturedLog log;
    const Library library = Library::Load({kMissing}, {kMissing});
    EXPECT_FALSE(library);
    EXPECT_EQ(library.Name(), "");
    EXPECT_EQ(library.Symbol(kLoadedSymbol), nullptr);
    EXPECT_TRUE(log.Messages().empty());
}

TEST(LibraryTest, ResolvesSymbolsByName) {
    const Library library = Library::Load({kLoaded});
    EXPECT_NE(library.Symbol(kLoadedSymbol), nullptr);
    EXPECT_EQ(library.Symbol("ernie_no_such_symbol"), nullptr);

    void* (*found)(size_t) = nullptr;
    void (*missing)() = nullptr;
    EXPECT_TRUE(library.Symbol(kLoadedSymbol, found));
    EXPECT_NE(found, nullptr);
    EXPECT_FALSE(library.Symbol("ernie_no_such_symbol", missing));
    EXPECT_EQ(missing, nullptr);
}

TEST(LibraryTest, SymbolResolverRemembersTheFirstMissingSymbol) {
    SymbolResolver resolve(Library::Load({kLoaded}));
    void* (*found)(size_t) = nullptr;
    void (*missing)() = nullptr;
    void (*alsoMissing)() = nullptr;
    resolve(kLoadedSymbol, found);
    EXPECT_EQ(resolve.Missing(), nullptr);
    resolve("ernie_no_such_symbol", missing);
    resolve("ernie_nor_this", alsoMissing);
    EXPECT_NE(found, nullptr);
    EXPECT_EQ(missing, nullptr);
    EXPECT_STREQ(resolve.Missing(), "ernie_no_such_symbol");
}

int Present() {
    return 1;
}

TEST(LibraryTest, SymbolResolverTakesAnyLookup) {
    SymbolResolver resolve([](const char* symbol) -> void* {
        return std::string_view(symbol) == "present" ? reinterpret_cast<void*>(&Present) : nullptr;
    });
    int (*present)() = nullptr;
    resolve("present", present);
    ASSERT_NE(present, nullptr);
    EXPECT_EQ(present(), 1);
    EXPECT_EQ(resolve.Missing(), nullptr);
}

TEST(LibraryTest, UnavailableApisAreLoggedAndRefused) {
    const test::CapturedLog log;
    const Loaded<int> loaded = LogUnavailable<int>("the runtime", 7);
    EXPECT_TRUE(log.Messages().empty());
    EXPECT_EQ(RequireLoaded(loaded), 7);

    const Loaded<int> failed = LogUnavailable<int>("the runtime", std::string("it is missing"));
    ASSERT_EQ(log.Messages().size(), 1u);
    EXPECT_EQ(log.Messages()[0].first, LogLevel::Debug);
    EXPECT_EQ(log.Messages()[0].second, "the runtime can't be used here: it is missing");
    try {
        RequireLoaded(failed);
        FAIL();
    } catch (const UnsupportedError& error) {
        EXPECT_STREQ(error.what(), "it is missing");
    }
}

TEST(LibraryTest, TooOldNamesBothVersions) {
    EXPECT_EQ(TooOld("the CUDA driver", Version{11, 8}, Version{12, 0}),
              "the CUDA driver is version 11.8; Ernie needs 12.0 or later");
}

}
}
