#pragma once

#include "sw_fwd.h"  // Forward declaration

#include <cstddef>  // std::nullptr_t

template <typename T, typename... Args>
SharedPtr<T> MakeShared(Args&&... args);

template <typename T>
class SharedPtr {
public:
    SharedPtr() = default;
    SharedPtr(std::nullptr_t){};
    template <typename U>
    explicit SharedPtr(U* ptr) : ptr_(ptr), ctrl_(new ControlBlock<U>(ptr)) {
        if constexpr (std::is_convertible_v<U*, ESTPBase*>) {
            ptr->weak_this_ = WeakPtr(*this);
        }
        ctrl_->strong++;
    }

    template <typename U>
    explicit SharedPtr(U* ptr, Control* ctrl) : ptr_(ptr), ctrl_(ctrl) {
        if constexpr (std::is_convertible_v<U*, ESTPBase*>) {
            ptr->weak_this_ = WeakPtr(*this);
        }
        ctrl_->strong++;
    }
    SharedPtr(const SharedPtr& other) : ctrl_(other.ctrl_), ptr_(other.ptr_) {
        if (ctrl_) {
            ctrl_->strong++;
        }
    };
    template <typename U>
    SharedPtr(const SharedPtr<U>& other) : ctrl_(other.ctrl_), ptr_(other.ptr_) {
        if (ctrl_) {
            ctrl_->strong++;
        }
    };
    SharedPtr(SharedPtr&& other) : ptr_(std::forward<T*>(other.ptr_)), ctrl_(other.ctrl_) {
        other.ctrl_ = nullptr;
        other.ptr_ = nullptr;
    };
    template <typename U>
    SharedPtr(SharedPtr<U>&& other) : ptr_(other.ptr_), ctrl_(other.ctrl_) {
        other.ctrl_ = nullptr;
        other.ptr_ = nullptr;
    };

    template <typename Y>
    SharedPtr(const SharedPtr<Y>& other, T* ptr) : ptr_(ptr), ctrl_(other.ctrl_) {
        if (ctrl_) {
            ctrl_->strong++;
        }
    };
    explicit SharedPtr(const WeakPtr<T>& other) {
        if (other.Expired()) {
            throw BadWeakPtr();
        } else {
            ctrl_ = other.GetControl();
            ptr_ = other.Get();
            if (ctrl_) {
                ctrl_->strong++;
            }
        }
    };

    SharedPtr& operator=(const SharedPtr& other) {
        if (this == &other) {
            return *this;
        }
        if (ctrl_) {
            ctrl_->DecStrong();
        }
        ctrl_ = other.ctrl_;
        ptr_ = other.ptr_;
        if (ctrl_) {
            ctrl_->strong++;
        }
        return *this;
    };
    template <typename U>
    SharedPtr& operator=(SharedPtr<U>&& other) {
        if (ctrl_) {
            ctrl_->DecStrong();
        }
        ctrl_ = other.ctrl_;
        ptr_ = other.ptr_;
        other.ctrl_ = nullptr;
        other.ptr_ = nullptr;
        return *this;
    };

    ~SharedPtr() {
        Reset();
    }

    void Reset() {
        if (ctrl_) {
            ctrl_->DecStrong();
            ctrl_ = nullptr;
        }
    };
    template <typename U>
    void Reset(U* ptr) {
        SharedPtr(ptr).Swap(*this);
    };
    void Swap(SharedPtr& other) {
        std::swap(ctrl_, other.ctrl_);
        std::swap(ptr_, other.ptr_);
    };

    T* Get() const {
        if (ctrl_) {
            return ptr_;
        }
        return nullptr;
    };
    Control* GetControl() const {
        return ctrl_;
    }
    T& operator*() const {
        if (ctrl_) {
            return *ptr_;
        }
    };
    T* operator->() const {
        if (ctrl_) {
            return ptr_;
        }
        return nullptr;
    };
    size_t UseCount() const {
        if (ctrl_) {
            return ctrl_->strong;
        }
        return 0;
    };
    explicit operator bool() const {
        return ptr_ != nullptr;
    };

private:
    template <typename Y>
    friend class SharedPtr;

    Control* ctrl_ = nullptr;

    T* ptr_ = nullptr;
};

template <typename T, typename U>
inline bool operator==(const SharedPtr<T>& left, const SharedPtr<U>& right) {
    return left.Get() == right.Get();
};

template <typename T, typename... Args>
SharedPtr<T> MakeShared(Args&&... args) {
    auto prom = new ControlBlockCringe<T>(std::forward<Args>(args)...);
    return SharedPtr<T>(prom->GetPtr(), prom);
}

// Look for usage examples in tests and seminar
template <typename T>
class EnableSharedFromThis : public ESTPBase {
public:
    SharedPtr<T> SharedFromThis() {
        return SharedPtr<T>(weak_this_);
    };
    SharedPtr<const T> SharedFromThis() const {
        return SharedPtr<const T>(weak_this_);
    };

    WeakPtr<T> WeakFromThis() noexcept {
        return WeakPtr<T>(weak_this_);
    };
    WeakPtr<const T> WeakFromThis() const noexcept {
        return WeakPtr<const T>(weak_this_);
    };
    WeakPtr<T> weak_this_;
    ~EnableSharedFromThis() = default;
};
