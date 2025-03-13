#pragma once

#include <exception>

class BadWeakPtr : public std::exception {};

template <typename T>
class SharedPtr;

template <typename T>
class WeakPtr;

struct Control {
    virtual ~Control() = default;
    virtual void DecStrong() = 0;
    virtual void DecWeak() = 0;
    int strong = 0;
    int weak = 0;
};