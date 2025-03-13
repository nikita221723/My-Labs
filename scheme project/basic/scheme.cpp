#include "scheme.h"
#include <functional>

Interpreter::Interpreter() {
    functions_["quote"] = std::make_shared<Object>();
    functions_["number?"] = std::make_shared<IsExpectedType<Number>>();
    functions_["="] = std::make_shared<Monotonic<std::equal_to<int64_t>>>();
    functions_[">"] = std::make_shared<Monotonic<std::greater<int64_t>>>();
    functions_[">="] = std::make_shared<Monotonic<std::greater_equal<int64_t>>>();
    functions_["<"] = std::make_shared<Monotonic<std::less<int64_t>>>();
    functions_["<="] = std::make_shared<Monotonic<std::less_equal<int64_t>>>();
    functions_["+"] = std::make_shared<IntegerArithmetics<std::plus<int64_t>>>();
    functions_["-"] = std::make_shared<IntegerArithmetics<std::minus<int64_t>>>();
    functions_["*"] = std::make_shared<IntegerArithmetics<std::multiplies<int64_t>>>();
    functions_["/"] = std::make_shared<IntegerArithmetics<std::divides<int64_t>>>();
    functions_["max"] = std::make_shared<IntegerMax>();
    functions_["min"] = std::make_shared<IntegerMin>();
    functions_["abs"] = std::make_shared<IntegerAbs>();
    functions_["boolean?"] = std::make_shared<IsBoolean>();
    functions_["not"] = std::make_shared<Not>();
    functions_["and"] = std::make_shared<And>();
    functions_["or"] = std::make_shared<Or>();
    functions_["pair?"] = std::make_shared<IsPair>();
    functions_["null?"] = std::make_shared<IsNull>();
    functions_["list?"] = std::make_shared<IsList>();
    functions_["cons"] = std::make_shared<Cons>();
    functions_["car"] = std::make_shared<Car>();
    functions_["cdr"] = std::make_shared<Cdr>();
    functions_["list"] = std::make_shared<ToList>();
    functions_["list-ref"] = std::make_shared<ListRef>();
    functions_["list-tail"] = std::make_shared<ListTail>();
}

std::shared_ptr<Object> Interpreter::ReadFull(const std::string& str) {
    std::stringstream ss{str};
    Tokenizer tokenizer{&ss};
    auto obj = Read(&tokenizer);
    if (!tokenizer.IsEnd()) {
        try {
            throw SyntaxError("");
        } catch (const SyntaxError& e) {
            throw;
        }
    }
    return obj;
}

std::string Interpreter::IWannaDie(const std::string& s) {
    std::string ans;
    std::stringstream ss{s};
    bool flag = false;
    Tokenizer tokenizer{&ss};
    tokenizer.Next();
    if (tokenizer.cur_token == "quote") {
        flag = true;
        tokenizer.Next();
    }
    while (!tokenizer.IsEnd()) {
        if (tokenizer.cur_token == "'" || tokenizer.cur_token == "quote") {
            try {
                throw RuntimeError("");
            } catch (const RuntimeError& e) {
                throw;
            }
        }
        if (tokenizer.cur_token == "(" || tokenizer.cur_token == ")" || (tokenizer.is_symbol)) {
            ans += tokenizer.cur_token;
        } else {
            if (ans.back() != '(') {
                ans += ' ';
                ans += tokenizer.cur_token;
            } else {
                ans += tokenizer.cur_token;
            }
        }
        tokenizer.Next();
    }
    if (flag) {
        ans.pop_back();
    }
    return ans;
}

std::string Interpreter::Run(const std::string& s) {
    auto current = ReadFull(s);
    if (!current) {
        try {
            throw RuntimeError("");
        } catch (const RuntimeError& e) {
            throw;
        }
    }
    auto prom = current->Evaluate(functions_);
    if (prom == nullptr) {
        return "()";
    }
    if (Is<Cell>(prom)) {
        As<Cell>(prom)->open = true;
    }
    if (current->is_unique) {
        current->is_unique = false;
        return ("(())");
    }
    return prom->ToString();
}
