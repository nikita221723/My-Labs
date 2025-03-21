#pragma once

#include "sw_fwd.h"  // Forward declaration

#include <cstddef>  // std::nullptr_t
template <typename T, typename... Args>
SharedPtr<T> MakeShared(Args&&... args);

struct Control {
    virtual ~Control() = default;
    virtual void DecStrong() = 0;
    int strong = 0;
};

template <typename T>
struct ControlBlock : public Control {
    ~ControlBlock() override = default;
    template <typename... Args>
    ControlBlock(T* ptr) : ptr_(ptr) {
    }
    void DecStrong() override {
        --strong;
        if (strong == 0) {
            delete ptr_;
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
    void DecStrong() {
        --strong;
        if (strong == 0) {
            GetPtr()->~T();
            delete this;
        }
    }
    T* GetPtr() {
        return reinterpret_cast<T*>(&store);
    }
    std::aligned_storage_t<sizeof(T), alignof(T)> store;
};

template <typename T>
class SharedPtr {
public:
    SharedPtr() = default;
    SharedPtr(std::nullptr_t){};
    template <typename U>
    explicit SharedPtr(U* ptr) : ptr_(ptr), ctrl_(new ControlBlock<U>(ptr)) {
        ctrl_->strong++;
    }

    template <typename U>
    explicit SharedPtr(U* ptr, ControlBlockCringe<U>* ctrl) : ptr_(ptr), ctrl_(ctrl) {
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
inline bool operator==(const SharedPtr<T>& left, const SharedPtr<U>& right);

template <typename T, typename... Args>
SharedPtr<T> MakeShared(Args&&... args) {
    auto prom = new ControlBlockCringe<T>(std::forward<Args>(args)...);
    return SharedPtr<T>(prom->GetPtr(), prom);
}

template <typename T>
class EnableSharedFromThis {
public:
    SharedPtr<T> SharedFromThis();
    SharedPtr<const T> SharedFromThis() const;

    WeakPtr<T> WeakFromThis() noexcept;
    WeakPtr<const T> WeakFromThis() const noexcept;
};
