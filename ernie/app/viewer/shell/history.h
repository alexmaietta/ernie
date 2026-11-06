// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ernie::app {

struct Edit {
    std::string name;
    std::function<void()> undo;
    std::function<void()> redo;
};

class History {
  public:
    static constexpr size_t kKept = 200;

    void Do(Edit edit) {
        edit.redo();
        Push(std::move(edit));
    }

    void Push(Edit edit) {
        m_undo.push_back(std::move(edit));
        if (m_undo.size() > kKept) {
            m_undo.erase(m_undo.begin());
        }
        m_redo.clear();
    }

    bool Undo() {
        if (m_undo.empty()) {
            return false;
        }
        Edit edit = std::move(m_undo.back());
        m_undo.pop_back();
        edit.undo();
        m_redo.push_back(std::move(edit));
        return true;
    }

    bool Redo() {
        if (m_redo.empty()) {
            return false;
        }
        Edit edit = std::move(m_redo.back());
        m_redo.pop_back();
        edit.redo();
        m_undo.push_back(std::move(edit));
        return true;
    }

    const std::string* NextUndo() const {
        return m_undo.empty() ? nullptr : &m_undo.back().name;
    }

    const std::string* NextRedo() const {
        return m_redo.empty() ? nullptr : &m_redo.back().name;
    }

    void Clear() {
        m_undo.clear();
        m_redo.clear();
    }

  private:
    std::vector<Edit> m_undo;
    std::vector<Edit> m_redo;
};

}
