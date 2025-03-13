#include <tokenizer.h>
SymbolToken::SymbolToken(std::string s) noexcept : name(s) {
}
bool SymbolToken::operator==(const SymbolToken &other) const {
    return name == other.name;
}

bool QuoteToken::operator==(const QuoteToken &) const {
    return true;
}
bool DotToken::operator==(const DotToken &) const {
    return true;
}

ConstantToken::ConstantToken(int v) noexcept : value(v) {
}
bool ConstantToken::operator==(const ConstantToken &other) const {
    return value == other.value;
}
Tokenizer::Tokenizer(std::istream *in) : stream(in) {
    Next();
}
void Tokenizer::Next() {
    cur_token = "";
    space = false;
    is_symbol = false;
    is_number = false;
    try {
        while (true) {
            if (stream->peek() == EOF) {
                if (cur_token.empty()) {
                    is_end = true;
                }
                break;
            }
            if (stream->peek() == ' ') {
                stream->get();
                space = true;
                if (!cur_token.empty()) {
                    break;
                }
                continue;
            }
            if (stream->peek() == '+' || stream->peek() == '-') {
                if (is_number) {
                    break;
                }
                cur_token.push_back(stream->get());
                if (stream->peek() == ' ') {
                    is_symbol = true;
                    continue;
                }
                if (stream->peek() >= 48 && stream->peek() <= 57) {
                    is_number = true;
                    continue;
                }
                is_symbol = true;
                continue;
            }
            if (stream->peek() >= 48 && stream->peek() <= 57) {
                if (!is_symbol) {
                    is_number = true;
                }
                if (!is_number && !is_symbol) {
                    break;
                }
                cur_token.push_back(stream->get());
                continue;
            }
            if (stream->peek() == '#') {
                if (is_number) {
                    break;
                }
                cur_token.push_back(stream->get());
                if (stream->peek() == 't' || stream->peek() == 'f') {
                    is_symbol = true;
                    cur_token.push_back(stream->get());
                    break;
                } else {
                    is_symbol = true;
                    break;
                }
            }
            if (stream->peek() == ')' || stream->peek() == '(' || stream->peek() == '.' ||
                stream->peek() == 39) {
                if (is_symbol || is_number) {
                    break;
                }
                cur_token.push_back(stream->get());
                break;
            }
            if (stream->peek() == 35 || stream->peek() == 42 || stream->peek() == 47 ||
                (stream->peek() >= 60 && stream->peek() <= 62) ||
                (stream->peek() >= 65 && stream->peek() <= 122)) {
                cur_token.push_back(stream->get());
                is_symbol = true;
                continue;
            }
            if (is_symbol) {
                if ((stream->peek() >= 48 && stream->peek() <= 57) || stream->peek() == 33 ||
                    stream->peek() == 63 || stream->peek() == 95) {
                    cur_token.push_back(stream->get());
                    continue;
                }
            }
            if (stream->peek() == '\n') {
                stream->get();
                continue;
            }
            throw SyntaxError("");
        }
    } catch (const SyntaxError &e) {
        throw;
    }
}
bool Tokenizer::IsEnd() {
    return is_end;
}
Token Tokenizer::GetToken() {
    if (is_number) {
        return ConstantToken{std::stoi(cur_token)};
    }
    if (is_symbol) {
        auto tok = SymbolToken{cur_token};
        if (cur_token == "#f" || cur_token == "#t") {
            tok.is_boolean = true;
        }
        return tok;
    }
    if (cur_token == "'") {
        return QuoteToken{};
    }
    if (cur_token == "(") {
        return BracketToken::OPEN;
    }
    if (cur_token == ")") {
        return BracketToken::CLOSE;
    }
    return DotToken{};
}