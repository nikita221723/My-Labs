#pragma once

#include "compressed_pair.h"

#include <cstddef>  // std::nullptr_t
#include <type_traits>

template <typename T>
struct Slug {
    Slug() = default;
    template <typename U>
    Slug(const Slug<U>&) {
    }
    void operator()(T* ptr) {
        static_assert(!std::is_void_v<T>);
        static_assert(sizeof(T) > 0);
        delete ptr;
    }
};

template <>
struct Slug<void> {
    void operator()(void* p) {
        int* x = reinterpret_cast<int*>(p);
        delete x;
    }
};

// Primary template
template <typename T, typename Deleter = Slug<T>>
class UniquePtr {
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors

    explicit UniquePtr(T* ptr = nullptr) noexcept : data_(ptr, Deleter()){};

    UniquePtr(T* ptr, Deleter deleter) noexcept : data_(ptr, std::forward<Deleter>(deleter)){};

    template <typename Del>
    UniquePtr(T* ptr, Del&& deleter) noexcept : data_(ptr, std::forward<Del>(deleter)){};

    UniquePtr(UniquePtr&& other) noexcept
        : data_(other.data_.GetFirst(), std::forward<Deleter>(other.data_.GetSecond())) {
        other.data_.GetFirst() = nullptr;
    };
    template <typename Derived, typename DerivedDeleter>
    UniquePtr(UniquePtr<Derived, DerivedDeleter>&& other) noexcept
        : data_(other.data_.GetFirst(), std::forward<DerivedDeleter>(other.data_.GetSecond())) {
        other.data_.GetFirst() = nullptr;
    };

    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(UniquePtr& rhs) = delete;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // `operator=`-s

    UniquePtr& operator=(UniquePtr&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        Reset();
        data_.GetFirst() = other.data_.GetFirst();
        data_.GetSecond() = std::move(other.data_.GetSecond());
        other.data_.GetFirst() = nullptr;
        return *this;
    };

    template <typename Derived, typename DerivedDeleter>
    UniquePtr& operator=(UniquePtr<Derived, DerivedDeleter>&& other) noexcept {
        if (this == reinterpret_cast<UniquePtr*>(&other)) {
            return *this;
        }
        Reset();
        data_.GetFirst() = other.data_.GetFirst();
        data_.GetSecond() = std::move(other.data_.GetSecond());
        other.data_.GetFirst() = nullptr;
        return *this;
    }
    UniquePtr& operator=(std::nullptr_t) noexcept {
        if (data_.GetFirst() != nullptr) {
            data_.GetSecond()(data_.GetFirst());
            data_.GetFirst() = nullptr;
        }
        return *this;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Destructor

    ~UniquePtr() {
        Reset();
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Modifiers

    T* Release() {
        auto prom = data_.GetFirst();
        if (data_.GetFirst() != nullptr) {
            data_.GetFirst() = nullptr;
        }
        return prom;
    }
    void Reset(T* ptr = nullptr) {
        auto prom = data_.GetFirst();
        data_.GetFirst() = ptr;
        if (prom != nullptr) {
            data_.GetSecond()(prom);
        }
    };
    void Swap(UniquePtr& other) {
        std::swap(data_, other.data_);
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Observers

    T* Get() const {
        return data_.GetFirst();
    };
    Deleter& GetDeleter() {
        return data_.GetSecond();
    };
    const Deleter& GetDeleter() const {
        return data_.GetSecond();
    };
    explicit operator bool() const {
        if (data_.GetFirst() == nullptr) {
            return false;
        }
        return true;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Single-object dereference operators

    T operator*() const {
        return *data_.GetFirst();
    };
    T* operator->() const {
        return data_.GetFirst();
    };
    CompressedPair<T*, Deleter> data_;
};

// Specialization for arrays

template <typename T>
struct Slug<T[]> {
    Slug() = default;
    void operator()(T* ptr) {
        delete[] ptr;
    }
};

template <typename T, typename Deleter>
class UniquePtr<T[], Deleter> {
public:
    explicit UniquePtr(T* ptr = nullptr) noexcept : data_(ptr, Deleter()){};

    UniquePtr(T* ptr, Deleter deleter) noexcept : data_(ptr, std::forward<Deleter>(deleter)){};

    template <typename Del>
    UniquePtr(T* ptr, Del&& deleter) noexcept : data_(ptr, std::forward<Del>(deleter)){};

    UniquePtr(UniquePtr&& other) noexcept
        : data_(other.data_.GetFirst(), std::forward<Deleter>(other.data_.GetSecond())) {
        other.data_.GetFirst() = nullptr;
    };

    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(UniquePtr& rhs) = delete;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // `operator=`-s

    UniquePtr& operator=(UniquePtr&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        Reset();
        data_.GetFirst() = other.data_.GetFirst();
        data_.GetSecond() = std::move(other.data_.GetSecond());
        other.data_.GetFirst() = nullptr;
        return *this;
    };

    UniquePtr& operator=(std::nullptr_t) noexcept {
        if (data_.GetFirst() != nullptr) {
            data_.GetSecond()(data_.GetFirst());
            data_.GetFirst() = nullptr;
        }
        return *this;
    };

    auto& operator[](int i) {
        return Get()[i];
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Destructor

    ~UniquePtr() {
        Reset();
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Modifiers

    T* Release() {
        auto prom = data_.GetFirst();
        if (data_.GetFirst() != nullptr) {
            data_.GetFirst() = nullptr;
        }
        return prom;
    }
    void Reset(T* ptr = nullptr) {
        auto prom = data_.GetFirst();
        data_.GetFirst() = ptr;
        if (prom != nullptr) {
            data_.GetSecond()(prom);
        }
    };
    void Swap(UniquePtr& other) {
        std::swap(data_, other.data_);
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Observers

    T* Get() const {
        return data_.GetFirst();
    };
    Deleter& GetDeleter() {
        return data_.GetSecond();
    };
    const Deleter& GetDeleter() const {
        return data_.GetSecond();
    };
    explicit operator bool() const {
        if (data_.GetFirst() == nullptr) {
            return false;
        }
        return true;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Single-object dereference operators

    T operator*() const {
        return *data_.GetFirst();
    };
    T* operator->() const {
        return data_.GetFirst();
    };

private:
    CompressedPair<T*, Deleter> data_;
};
