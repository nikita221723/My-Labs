#pragma once

#include <exception>
class ESTPBase {};
template <typename T>
class EnableSharedFromThis;

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
template <typename T>
struct ControlBlock : public Control {
    ControlBlock(T* ptr) : ptr_(ptr) {
    }
    void DecStrong() override {
        --strong;
        if (strong == 0 && weak == 0) {
            auto prom = ptr_;
            if (prom) {
                delete prom;
            }
            ptr_ = nullptr;
            delete this;
        } else if (strong == 0) {
            if constexpr (std::is_convertible_v<T*, ESTPBase*>) {
                if (ptr_->weak_this_.ctrl_) {
                    ptr_->weak_this_.ctrl_->DecWeak();
                }
                return;
            }
            auto prom = ptr_;
            if (prom) {
                delete prom;
            }
            ptr_ = nullptr;
        }
    }
    void DecWeak() override {
        --weak;
        if (strong == 0 && weak == 0) {
            if constexpr (std::is_convertible_v<T*, ESTPBase*>) {
                delete ptr_;
            }
            delete this;
        }
    }
    T* ptr_ = nullptr;
};
template <typename T>
struct ControlBlockCringe : public Control {
    ~ControlBlockCringe() override = default;
    template <typename... Args>
    ControlBlockCringe(Args&&... args) {
        new (&store) T(std::forward<Args>(args)...);
    }
    void DecStrong() override {
        --strong;
        if (strong == 0 && weak == 0) {
            GetPtr()->~T();
            delete this;
        } else if (strong == 0) {
            GetPtr()->~T();
        }
    }
    void DecWeak() override {
        --weak;
        if (strong == 0 && weak == 0) {
            delete this;
        }
    }
    T* GetPtr() {
        return reinterpret_cast<T*>(&store);
    }
    std::aligned_storage_t<sizeof(T), alignof(T)> store;
};