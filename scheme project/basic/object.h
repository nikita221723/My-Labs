#pragma once

#include <utility>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include "error.h"
class Object : public std::enable_shared_from_this<Object> {
public:
    virtual ~Object() = default;
    virtual std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>>) {
        try {
            throw RuntimeError("");
        } catch (const RuntimeError& e) {
            throw;
        }
    };
    virtual std::shared_ptr<Object> Evaluate(
        std::unordered_map<std::string, std::shared_ptr<Object>>&) {
        try {
            throw RuntimeError("");
        } catch (const RuntimeError& e) {
            throw;
        }
    };
    virtual std::string ToString() {
        try {
            throw RuntimeError("");
        } catch (const RuntimeError& e) {
            throw;
        }
    };

    bool is_unique = false;
};

static std::vector<std::shared_ptr<Object>> Helper(
    std::shared_ptr<Object>, std::unordered_map<std::string, std::shared_ptr<Object>>&);

template <typename T>
std::shared_ptr<T> As(const std::shared_ptr<Object>& obj) {
    return std::dynamic_pointer_cast<T>(obj);
}

template <typename T>
bool Is(const std::shared_ptr<Object>& obj) {
    return typeid(*obj.get()) == typeid(T);
}

class Number : public Object {
public:
    Number(int64_t v) : val_(v){};
    int GetValue() const {
        return val_;
    }
    std::shared_ptr<Object> Evaluate(
        std::unordered_map<std::string, std::shared_ptr<Object>>&) override {
        return std::make_shared<Number>(Number(GetValue()));
    };
    std::string ToString() override {
        return std::to_string(GetValue());
    }

private:
    int64_t val_ = 0;
};

class Symbol : public Object {
public:
    const std::string& GetName() const {
        return name_;
    }
    Symbol(std::string name) : name_(name) {
    }
    std::string ToString() override {
        return GetName();
    }
    std::shared_ptr<Object> Evaluate(
        std::unordered_map<std::string, std::shared_ptr<Object>>&) override {
        if (name_ == "#f" || name_ == "#t") {
            return std::make_shared<Symbol>(Symbol(name_));
        }
        try {
            throw RuntimeError("");
        } catch (const RuntimeError& e) {
            throw;
        }
    };

private:
    std::string name_;
};

class Cell : public Object {
public:
    Cell(std::shared_ptr<Object> first, std::shared_ptr<Object> second) : pair_({first, second}){};
    std::shared_ptr<Object> GetFirst() const {
        return pair_.first;
    }
    std::shared_ptr<Object> GetSecond() const {
        return pair_.second;
    }
    std::string ToString() override {
        std::string ans;
        std::string real_ans;
        if (open) {
            ans.push_back('(');
            open = false;
        }
        if (GetSecond() != nullptr && !(Is<Cell>(GetSecond()))) {
            ans += GetFirst()->ToString();
            ans += " . ";
            ans += GetSecond()->ToString();
            if (ans[0] == '(') {
                ans.push_back(')');
            }
            return ans;
        }
        if (GetFirst()) {
            if (Is<Cell>(GetFirst())) {
                for (auto a : GetFirst()->ToString()) {
                    ans.push_back(a);
                }
            } else {
                ans += GetFirst()->ToString();
                ans.push_back(' ');
            }
        }
        if (GetSecond()) {
            if (Is<Cell>(GetSecond())) {
                for (auto a : GetSecond()->ToString()) {
                    ans.push_back(a);
                }
            } else {
                ans += GetSecond()->ToString();
                ans.push_back(' ');
            }
        }
        if (ans[0] == '(') {
            ans.push_back(')');
        }
        for (auto i : ans) {
            if (i != ' ' && i != '(' && i != ')') {
                real_ans.push_back(i);
                real_ans.push_back(' ');
            }
        }
        if (!real_ans.empty()) {
            real_ans.pop_back();
        }
        if (ans[0] == '(' && ans.back() == ')') {
            real_ans.insert(0, "(");
            real_ans.push_back(')');
        }
        return real_ans;
    }
    std::shared_ptr<Object> Evaluate(
        std::unordered_map<std::string, std::shared_ptr<Object>>& mapa) override {
        std::shared_ptr<Object> func = GetFirst();
        if (!func) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        if (typeid(*func.get()) != typeid(Symbol)) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        auto func_name = std::dynamic_pointer_cast<Symbol>(func)->GetName();
        if (func_name == "quote") {
            auto applied = GetSecond();
            if (applied == nullptr) {
                try {
                    throw RuntimeError("");
                } catch (const RuntimeError& e) {
                    throw;
                }
            }
            if (Is<Cell>(applied)) {
                if (As<Cell>(applied)->GetSecond() == nullptr) {
                    if (As<Cell>(applied)->GetFirst()) {
                        if (Is<Cell>(As<Cell>(applied)->GetFirst())) {
                            auto pr = As<Cell>(As<Cell>(applied)->GetFirst());
                            if (!pr->GetSecond() && !pr->GetFirst()) {
                                is_unique = true;
                            }
                        }
                    }
                } else if (As<Cell>(applied)->GetFirst() == nullptr) {
                    if (Is<Cell>(As<Cell>(applied)->GetSecond())) {
                        auto pr = As<Cell>(As<Cell>(applied)->GetSecond());
                        if (!pr->GetSecond() && !pr->GetFirst()) {
                            try {
                                throw RuntimeError("");
                            } catch (const RuntimeError& e) {
                                throw;
                            }
                        }
                    }
                }
            }
            return applied;
        }
        if (mapa.find(func_name) == mapa.end()) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        auto applied = mapa[func_name]->Apply(Helper(GetSecond(), mapa));
        return applied;
    }
    inline static bool open = true;

private:
    std::pair<std::shared_ptr<Object>, std::shared_ptr<Object>> pair_;
};

static std::vector<std::shared_ptr<Object>> Helper2(std::shared_ptr<Cell> v);

template <typename T>
class IsExpectedType : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        for (auto obj : v) {
            if (typeid(*obj.get()) != typeid(T)) {
                return std::make_shared<Symbol>(Symbol("#f"));
            }
        }
        return std::make_shared<Symbol>(Symbol("#t"));
    }
};
template <typename F>
class Monotonic : public Object {
    F comparator_;

public:
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        if (v.size() < 2) {
            return std::make_shared<Symbol>(Symbol("#t"));
        }
        auto first = std::dynamic_pointer_cast<Number>(v[0]);
        if (!first) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }

        for (auto i = 1; i < v.size(); ++i) {
            auto prom = std::dynamic_pointer_cast<Number>(v[i]);
            if (!prom) {
                try {
                    throw RuntimeError("");
                } catch (const RuntimeError& e) {
                    throw;
                }
            }
            if (!comparator_(first->GetValue(), prom->GetValue())) {
                return std::make_shared<Symbol>(Symbol("#f"));
            }
        }
        return std::make_shared<Symbol>(Symbol("#t"));
    }
};
template <typename F>
class IntegerArithmetics : public Object {
    F func_;

public:
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        if (v.empty()) {
            if (func_(1, 2) == 2) {
                return std::make_shared<Number>(Number(1));
            }
            if (func_(1, 2) == 3) {
                return std::make_shared<Number>(Number(0));
            }
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        auto first = std::dynamic_pointer_cast<Number>(v[0]);
        if (!first) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        auto ans = first->GetValue();
        for (auto i = 1; i < v.size(); ++i) {
            auto prom = std::dynamic_pointer_cast<Number>(v[i]);
            if (!prom) {
                try {
                    throw RuntimeError("");
                } catch (const RuntimeError& e) {
                    throw;
                }
            }
            ans = func_(ans, prom->GetValue());
        }
        return std::make_shared<Number>(Number(ans));
    }
};

class IntegerMax : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        if (v.empty()) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        auto first = std::dynamic_pointer_cast<Number>(v[0]);
        if (!first) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        auto ans = first->GetValue();
        for (auto i = 1; i < v.size(); ++i) {
            auto prom = std::dynamic_pointer_cast<Number>(v[i]);
            if (!prom) {
                try {
                    throw RuntimeError("");
                } catch (const RuntimeError& e) {
                    throw;
                }
            }
            if (prom->GetValue() > ans) {
                ans = prom->GetValue();
            }
        }
        return std::make_shared<Number>(Number(ans));
    }
};
class IntegerMin : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        if (v.empty()) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        auto first = std::dynamic_pointer_cast<Number>(v[0]);
        if (!first) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        auto ans = first->GetValue();
        for (auto i = 1; i < v.size(); ++i) {
            auto prom = std::dynamic_pointer_cast<Number>(v[i]);
            if (!prom) {
                try {
                    throw RuntimeError("");
                } catch (const RuntimeError& e) {
                    throw;
                }
            }
            if (prom->GetValue() < ans) {
                ans = prom->GetValue();
            }
        }
        return std::make_shared<Number>(Number(ans));
    }
};
class IntegerAbs : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        if (v.size() != 1) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        auto first = std::dynamic_pointer_cast<Number>(v[0]);
        if (!first) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        } else {
            return std::make_shared<Number>(Number(std::abs(first->GetValue())));
        }
    }
};
class IsBoolean : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        if (v.size() != 1) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        auto first = std::dynamic_pointer_cast<Symbol>(v[0]);
        if (!first) {
            return std::make_shared<Symbol>(Symbol("#f"));
        }
        if (first->GetName() != "#f" && first->GetName() != "#t") {
            return std::make_shared<Symbol>(Symbol("#f"));
        }
        return std::make_shared<Symbol>(Symbol("#t"));
    }
};
class Not : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        if (v.size() != 1) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        if (v[0] == nullptr) {
            return std::make_shared<Symbol>(Symbol("#t"));
        }
        auto first = std::dynamic_pointer_cast<Symbol>(v[0]);
        if (!first) {
            return std::make_shared<Symbol>(Symbol("#f"));
        }
        if (first->GetName() == "#f") {
            return std::make_shared<Symbol>(Symbol("#t"));
        }
        return std::make_shared<Symbol>(Symbol("#f"));
    }
};
class And : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        std::vector<std::shared_ptr<Object>> my;
        for (auto val : v) {
            if (Is<Cell>(val)) {
                for (auto fuck : Helper2(As<Cell>(val))) {
                    my.push_back(fuck);
                }
            } else {
                my.push_back(val);
            }
        }
        if (v.empty()) {
            return std::make_shared<Symbol>(Symbol("#t"));
        }
        std::vector<int> to_multiply;
        for (auto obj : my) {
            if (!obj) {
                to_multiply.push_back(0);
            } else if (Is<Symbol>(obj)) {
                if (!(As<Symbol>(obj)->GetName() == "#f")) {
                    to_multiply.push_back(1);
                } else {
                    to_multiply.push_back(0);
                }
            } else {
                to_multiply.push_back(1);
            }
        }
        int ans = 1;
        for (auto i = 0; i < to_multiply.size(); ++i) {
            ans *= to_multiply[i];
            if (ans == 0) {
                return my[i];
            }
        }
        return my.back();
    }
};
class Or : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        std::vector<std::shared_ptr<Object>> my;
        for (auto val : v) {
            if (Is<Cell>(val)) {
                for (auto fuck : Helper2(As<Cell>(val))) {
                    my.push_back(fuck);
                }
            } else {
                my.push_back(val);
            }
        }
        if (v.empty()) {
            return std::make_shared<Symbol>(Symbol("#f"));
        }
        std::vector<int> to_sum;
        for (auto obj : my) {
            if (!obj) {
                to_sum.push_back(0);
            } else if (Is<Symbol>(obj)) {
                if (!(As<Symbol>(obj)->GetName() == "#f")) {
                    to_sum.push_back(1);
                } else {
                    to_sum.push_back(0);
                }
            } else {
                to_sum.push_back(1);
            }
        }
        int ans = 0;
        for (auto i = 0; i < to_sum.size(); ++i) {
            ans += to_sum[i];
            if (ans != 0) {
                return my[i];
            }
        }
        return my.back();
    }
};
class IsPair : public Object {
public:
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        if (v.empty()) {
            return std::make_shared<Symbol>(Symbol("#f"));
        }
        auto p = v[0];
        if (!Is<Cell>(p)) {
            return std::make_shared<Symbol>(Symbol("#f"));
        }
        auto to_work = As<Cell>(p)->GetFirst();
        if (!to_work) {
            return std::make_shared<Symbol>(Symbol("#f"));
        }
        if (!Is<Cell>(to_work)) {
            return std::make_shared<Symbol>(Symbol("#f"));
        }
        if (!Is<Cell>(As<Cell>(to_work)->GetFirst()) && !Is<Cell>(As<Cell>(to_work)->GetSecond())) {
            return std::make_shared<Symbol>(Symbol("#t"));
        } else if (!Is<Cell>(As<Cell>(to_work)->GetFirst()) &&
                   Is<Cell>(As<Cell>(to_work)->GetSecond())) {
            auto last = As<Cell>(As<Cell>(to_work)->GetSecond());
            if (last->GetSecond() == nullptr && !Is<Cell>(last->GetFirst())) {
                return std::make_shared<Symbol>(Symbol("#t"));
            }
        }
        return std::make_shared<Symbol>(Symbol("#f"));
    }
};
class IsNull : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        auto p = v[0];
        auto to_work = As<Cell>(p)->GetFirst();
        if (!to_work) {
            return std::make_shared<Symbol>(Symbol("#t"));
        }
        return std::make_shared<Symbol>(Symbol("#f"));
    }
};
class IsList : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        auto p = v[0]->ToString();
        for (auto s : p) {
            if (s == '.') {
                return std::make_shared<Symbol>(Symbol("#f"));
            }
        }
        return std::make_shared<Symbol>(Symbol("#t"));
    }
};
class Cons : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        if (v.size() != 2) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        return std::make_shared<Cell>(Cell(v[0], v[1]));
    }
};
class Car : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        if (v.size() != 1) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        auto prom = As<Cell>(v[0])->GetFirst();
        if (!prom) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        return As<Cell>(prom)->GetFirst();
    }
};
class Cdr : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        if (v.size() != 1) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        auto prom = As<Cell>(v[0])->GetFirst();
        if (!prom) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        return As<Cell>(prom)->GetSecond();
    }
};
class ToList : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        if (v.empty()) {
            return nullptr;
        }
        auto first = v[0];
        std::vector<std::shared_ptr<Object>> new_vec;
        for (auto i = 1; i < v.size(); ++i) {
            new_vec.push_back(v[i]);
        }
        ToList t;
        if (new_vec.empty()) {
            return std::make_shared<Cell>(Cell(first, nullptr));
        }
        return std::make_shared<Cell>(Cell(first, t.Apply(new_vec)));
    }
};
class ListRef : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        int64_t r = 0;
        if (!Is<Number>(As<Cell>(As<Cell>(v[0])->GetSecond())->GetFirst())) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        } else {
            r = As<Number>((As<Cell>(As<Cell>(v[0])->GetSecond())->GetFirst()))->GetValue();
        }
        auto first = As<Cell>(v[0])->GetFirst();
        int cnt = 0;
        std::shared_ptr<Object> prom = first;
        while (true) {
            if (!prom) {
                try {
                    throw RuntimeError("");
                } catch (const RuntimeError& e) {
                    throw;
                }
            }
            if (cnt == r) {
                return As<Cell>(prom)->GetFirst();
            }
            prom = As<Cell>(prom)->GetSecond();
            ++cnt;
        }
    }
};
class ListTail : public Object {
    std::shared_ptr<Object> Apply(std::vector<std::shared_ptr<Object>> v) override {
        int64_t r = 0;
        if (!Is<Number>(As<Cell>(As<Cell>(v[0])->GetSecond())->GetFirst())) {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        } else {
            r = As<Number>((As<Cell>(As<Cell>(v[0])->GetSecond())->GetFirst()))->GetValue();
        }
        auto first = As<Cell>(v[0])->GetFirst();
        int cnt = 0;
        std::shared_ptr<Object> prom = first;
        while (true) {
            if (!prom) {
                try {
                    throw RuntimeError("");
                } catch (const RuntimeError& e) {
                    throw;
                }
            }
            if (cnt + 1 == r) {
                return As<Cell>(prom)->GetSecond();
            }
            prom = As<Cell>(prom)->GetSecond();
            ++cnt;
        }
    }
};
std::vector<std::shared_ptr<Object>> Helper(
    std::shared_ptr<Object> obj, std::unordered_map<std::string, std::shared_ptr<Object>>& mapa) {
    std::vector<std::shared_ptr<Object>> ans;
    auto cell = std::dynamic_pointer_cast<Cell>(obj);
    if (cell == nullptr) {
        return ans;
    }
    if (cell->GetFirst() != nullptr) {
        if (Is<Number>(cell->GetFirst())) {
            ans.push_back(cell->GetFirst()->Evaluate(mapa));
        } else if (Is<Symbol>(cell->GetFirst())) {
            if (mapa.find(std::dynamic_pointer_cast<Symbol>(cell->GetFirst())->GetName()) !=
                mapa.end()) {
                ans.push_back(cell->Evaluate(mapa));
                return ans;
            } else {
                auto s = std::dynamic_pointer_cast<Symbol>(cell->GetFirst());
                ans.push_back(s);
            }
        } else if (Is<Cell>(cell->GetFirst())) {
            for (auto pr : Helper(cell->GetFirst(), mapa)) {
                ans.push_back(pr);
            }
        }
    } else {
        ans.push_back(nullptr);
    }
    if (cell->GetSecond() != nullptr) {
        if (Is<Number>(cell->GetSecond())) {
            ans.push_back(cell->GetSecond()->Evaluate(mapa));
        } else if (Is<Symbol>(cell->GetSecond())) {
            if (mapa.find(std::dynamic_pointer_cast<Symbol>(cell->GetSecond())->GetName()) !=
                mapa.end()) {
                ans.push_back(cell->Evaluate(mapa));
                return ans;
            } else {
                auto s = std::dynamic_pointer_cast<Symbol>(cell->GetSecond());
                ans.push_back(s);
            }
        } else if (Is<Cell>(cell->GetSecond())) {
            for (auto pr : Helper(cell->GetSecond(), mapa)) {
                ans.push_back(pr);
            }
        }
    }
    return ans;
}
std::vector<std::shared_ptr<Object>> Helper2(std::shared_ptr<Cell> v) {
    std::vector<std::shared_ptr<Object>> ans;
    ans.push_back(v->GetFirst());
    if (Is<Cell>(v->GetSecond())) {
        std::shared_ptr<Cell> cur = As<Cell>(v->GetSecond());
        if (Is<Symbol>(cur->GetFirst()) && As<Symbol>(cur->GetFirst())->GetName() == "quote") {
            ans.push_back(cur->GetSecond());
        }
    }
    return ans;
}