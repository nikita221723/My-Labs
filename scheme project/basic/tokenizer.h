#pragma once

#include <variant>
#include <optional>
#include <istream>
#include "error.h"

struct SymbolToken {
    std::string name;
    SymbolToken(std::string s) noexcept;
    bool operator==(const SymbolToken& other) const;
    bool is_boolean = false;
};

struct QuoteToken {
    bool operator==(const QuoteToken&) const;
};

struct DotToken {
    bool operator==(const DotToken&) const;
};

enum class BracketToken { OPEN, CLOSE };

struct ConstantToken {
    ConstantToken(int v) noexcept;
    int value;

    bool operator==(const ConstantToken& other) const;
};

using Token = std::variant<ConstantToken, BracketToken, SymbolToken, QuoteToken, DotToken>;

class Tokenizer {
public:
    Tokenizer(std::istream* in);

    bool IsEnd();

    void Next();

    Token GetToken();
    std::string cur_token;
    std::istream* stream;
    bool is_end = false;
    bool space = false;
    bool is_symbol = false;
    bool is_number = false;
};