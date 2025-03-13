#pragma once

#include "sw_fwd.h"  // Forward declaration

// https://en.cppreference.com/w/cpp/memory/weak_ptr
template <typename T>
class WeakPtr {
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors

    WeakPtr() = default;

    WeakPtr(const WeakPtr& other) : ctrl_(other.ctrl_), ptr_(other.ptr_) {
        if (ctrl_) {
            ctrl_->weak++;
        }
    };

    template <typename U>
    WeakPtr(const WeakPtr<U>& other) : ctrl_(other.ctrl_), ptr_(other.ptr_) {
        if (ctrl_) {
            ctrl_->weak++;
        }
    };
    WeakPtr(WeakPtr&& other) : ctrl_(other.ctrl_), ptr_(std::forward<T*>(other.ptr_)) {
        other.ctrl_ = nullptr;
        other.ptr_ = nullptr;
    };

    template <typename U>
    WeakPtr(WeakPtr<U>&& other) : ctrl_(other.ctrl_), ptr_(other.ptr_) {
        other.ctrl_ = nullptr;
        other.ptr_ = nullptr;
    };

    // Demote `SharedPtr`
    // #2 from https://en.cppreference.com/w/cpp/memory/weak_ptr/weak_ptr
    WeakPtr(const SharedPtr<T>& other) : ptr_(other.Get()), ctrl_(other.GetControl()) {
        if (ctrl_) {
            ctrl_->weak++;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // `operator=`-s

    WeakPtr& operator=(const WeakPtr& other) {
        if (this == &other) {
            return *this;
        }
        if (ctrl_) {
            ctrl_->DecWeak();
        }
        ctrl_ = other.ctrl_;
        if (ctrl_) {
            ctrl_->weak++;
        }
        ptr_ = other.ptr_;
        return *this;
    };

    WeakPtr& operator=(WeakPtr&& other) {
        if (ctrl_) {
            ctrl_->DecWeak();
        }
        ctrl_ = other.ctrl_;
        ptr_ = other.ptr_;
        other.ctrl_ = nullptr;
        other.ptr_ = nullptr;
        return *this;
    };

    template <typename U>
    WeakPtr& operator=(WeakPtr<U>&& other) {
        if (ctrl_) {
            ctrl_->DecWeak();
        }
        ctrl_ = other.ctrl_;
        ptr_ = other.ptr_;
        other.ctrl_ = nullptr;
        other.ptr_ = nullptr;
        return *this;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Destructor

    ~WeakPtr() {
        if (ctrl_) {
            ctrl_->DecWeak();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Modifiers

    void Reset() {
        if (ctrl_) {
            ctrl_->DecWeak();
            ctrl_ = nullptr;
        }
    };
    void Swap(WeakPtr& other) {
        std::swap(ctrl_, other.ctrl_);
        std::swap(ptr_, other.ptr_);
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Observers

    size_t UseCount() const {
        if (ctrl_) {
            return ctrl_->strong;
        }
        return 0;
    };
    bool Expired() const {
        return UseCount() == 0;
    };
    SharedPtr<T> Lock() const {
        if (Expired()) {
            return {};
        }
        return SharedPtr<T>(ptr_, ctrl_);
    };
    Control* GetControl() const {
        return ctrl_;
    }
    T* Get() const {
        return ptr_;
    }

    template <typename U>
    friend class WeakPtr;

    Control* ctrl_ = nullptr;
    T* ptr_ = nullptr;
};
