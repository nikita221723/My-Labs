#pragma once

#include <type_traits>
#include <utility>

// Me think, why waste time write lot code, when few code do trick.
template <typename F, typename S, bool = std::is_empty_v<F> && !std::is_final_v<F>,
          bool = std::is_empty_v<S> && !std::is_final_v<S>, bool = std::is_same_v<F, S>>
class CompressedPair;

template <typename F, typename S>
class CompressedPair<F, S, true, true, false> : public F, public S {
public:
    CompressedPair() {
        F();
        S();
    }
    template <typename FF, typename SS>
    CompressedPair(FF&& first_r, SS&& second_r) {
        F(std::forward<FF>(first_r));
        S(std::forward<SS>(second_r));
    };
    F& GetFirst() {
        return *this;
    }
    const F& GetFirst() const {
        return static_cast<const F&>(*this);
    }
    S& GetSecond() {
        return *this;
    };
    const S& GetSecond() const {
        return static_cast<const S&>(*this);
    };
};

template <typename F, typename S>
class CompressedPair<F, S, false, true, false> : public S {
public:
    CompressedPair() {
        first_ = F();
        S();
    }
    template <typename FF, typename SS>
    CompressedPair(FF&& first_r, SS&& second_r) {
        first_ = std::forward<FF>(first_r);
        S(std::forward<SS>(second_r));
    }
    F& GetFirst() {
        return first_;
    }
    const F& GetFirst() const {
        return static_cast<const F&>(first_);
    }
    S& GetSecond() {
        return *this;
    };
    const S& GetSecond() const {
        return static_cast<const S&>(*this);
    };
    F first_;
};

template <typename F, typename S>
class CompressedPair<F, S, true, false, false> : public F {
public:
    CompressedPair() {
        second_ = S();
        F();
    }
    template <typename FF, typename SS>
    CompressedPair(FF&& first_r, S&& second_r) {
        second_ = std::forward<SS>(second_r);
        F(std::forward<FF>(first_r));
    }
    F& GetFirst() {
        return *this;
    }
    const F& GetFirst() const {
        return static_cast<const F&>(*this);
    }
    S& GetSecond() {
        return second_;
    };
    const S& GetSecond() const {
        return static_cast<const S&>(second_);
    };
    S second_;
};

template <typename F, typename S>
class CompressedPair<F, S, false, false, false> {
public:
    CompressedPair() {
        first_ = F();
        second_ = S();
    };
    template <typename FF, typename SS>
    CompressedPair(FF&& first, SS&& second)
        : first_(std::forward<FF>(first)), second_(std::forward<SS>(second)){};
    F& GetFirst() {
        return first_;
    }
    const F& GetFirst() const {
        return static_cast<const F&>(first_);
    }
    S& GetSecond() {
        return second_;
    };
    const S& GetSecond() const {
        return second_;
    };

private:
    F first_;
    S second_;
};

template <typename F, typename S>
class CompressedPair<F, S, false, false, true> {
public:
    CompressedPair() {
        first_ = F();
        second_ = S();
    };
    template <typename FF, typename SS>
    CompressedPair(FF&& first, SS&& second)
        : first_(std::forward<FF>(first)), second_(std::forward<SS>(second)){};
    F& GetFirst() {
        return first_;
    }
    const F& GetFirst() const {
        return static_cast<const F&>(first_);
    }
    S& GetSecond() {
        return second_;
    };
    const S& GetSecond() const {
        return second_;
    };

private:
    F first_;
    S second_;
};
template <typename F, typename S>
class CompressedPair<F, S, true, true, true> : public F {
public:
    CompressedPair() {
        F();
        second_ = S();
    }
    template <typename FF, typename SS>
    CompressedPair(FF&& first_r, SS&& second_r) {
        F(std::forward<FF>(first_r));
        second_(std::forward<SS>(second_r));
    };
    F& GetFirst() {
        return *this;
    }
    const F& GetFirst() const {
        return static_cast<const F&>(*this);
    }
    S& GetSecond() {
        return second_;
    };
    const S& GetSecond() const {
        return static_cast<const S&>(second_);
    };

private:
    S second_;
};