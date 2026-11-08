// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "ernie/export.h"
#include "ernie/scene/schema/ids.h"
#include "ernie/scene/schema/schema.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/view.h"
#include "ernie/util/span.h"

namespace ernie {

class Scene;

struct Connection {
    std::string input;
    std::string source;
    std::string output;
};

class ERNIE_API Prim {
  public:
    Prim() = default;

    Prim(Scene& scene, uint32_t index, uint32_t generation)
        : m_scene(&scene), m_index(index), m_generation(generation) {}

    bool Valid() const;

    uint32_t Index() const {
        return m_index;
    }

    uint32_t Generation() const {
        return m_generation;
    }

    SchemaId Schema() const;
    uint32_t World() const;
    bool IsSession() const;
    std::string Path() const;
    std::string_view Name() const;
    std::optional<Prim> Parent() const;
    std::vector<Prim> Children() const;
    std::vector<Connection> Connections() const;
    void Connect(std::string_view input, const Prim& source, std::string_view output) const;
    void Disconnect(std::string_view input) const;
    bool IsA(SchemaId schema) const;
    bool HasApi(SchemaId api) const;
    std::vector<std::string> Instances(SchemaId api) const;

    template <class T> bool IsA() const {
        return IsA(T::kSchema);
    }

    template <class T> bool HasApi() const {
        return HasApi(T::kSchema);
    }

    template <class T> std::vector<std::string> Instances() const {
        return Instances(T::kSchema);
    }

    template <class T> T Get(Field<T> field, std::string_view instance) const {
        return Read(Single(field.owner, instance), field);
    }

    template <class T>
    void Set(Field<T> field, std::string_view instance,
             const typename Identity<T>::Type& value) const {
        Write(Single(field.owner, instance), field, value);
    }

    Scene& GetScene() const;

    template <class T> T Get(std::string_view field) const {
        return Get<T>(Single().Column(field));
    }

    template <class T> void Set(std::string_view field, const T& value) const {
        Set(Single().Column(field), value);
    }

    template <class T> T Get(Field<T> field) const {
        return Read(Single(), field);
    }

    template <class T> void Set(Field<T> field, const typename Identity<T>::Type& value) const {
        Write(Single(), field, value);
    }

    bool operator==(const Prim& other) const {
        return m_scene == other.m_scene && m_index == other.m_index &&
               m_generation == other.m_generation;
    }

    bool operator!=(const Prim& other) const {
        return !(*this == other);
    }

  protected:
    void CheckSchema(SchemaId schema) const;

  private:
    template <class T> T Get(ColumnRef column) const {
        return Single().Get<T>(column).Get().front();
    }

    template <class T> void Set(ColumnRef column, const T& value) const {
        Single().Get<T>(column).Set(value);
    }

    template <class T> T Read(const View& view, Field<T> field) const {
        const ColumnRef column = field.Column();
        if constexpr (std::is_same_v<T, std::string>) {
            return std::string(Text(view.Get<Token>(column).Get().front()));
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            const UntypedAttribute values = view.Get(column);
            return column.column->type == ColumnType::TokenArray ? values.GetTokens(0)
                                                                 : values.GetTargets(0);
        } else if constexpr (std::is_same_v<T, std::vector<Primvar>>) {
            return view.Get(column).GetPrimvars(0);
        } else if constexpr (HasItems<T>::value) {
            return view.GetArray<typename T::value_type>(column).Get(0);
        } else {
            return view.Get<T>(column).Get().front();
        }
    }

    template <class T>
    void Write(const View& view, Field<T> field, const typename Identity<T>::Type& value) const {
        const ColumnRef column = field.Column();
        if constexpr (std::is_same_v<T, std::string>) {
            view.Get<Token>(column).Set(Intern(value));
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            UntypedAttribute values = view.Get(column);
            if (column.column->type == ColumnType::TokenArray) {
                values.SetTokens(0, value);
            } else {
                values.SetTargets(0, value);
            }
        } else if constexpr (std::is_same_v<T, std::vector<Primvar>>) {
            view.Get(column).SetPrimvars(0, value);
        } else if constexpr (HasItems<T>::value) {
            view.GetArray<typename T::value_type>(column).Set(
                0, Span<const typename T::value_type>(value));
        } else {
            view.Get<T>(column).Set(value);
        }
    }

    void Check() const;
    View Single() const;
    View Single(SchemaId api, std::string_view instance) const;
    Token Intern(std::string_view text) const;
    std::string_view Text(Token token) const;

    Scene* m_scene = nullptr;
    uint32_t m_index = kNoPrim;
    uint32_t m_generation = 0;
};

}
