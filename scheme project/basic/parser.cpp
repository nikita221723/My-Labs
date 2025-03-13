#include <parser.h>
#include <vector>

std::shared_ptr<Object> Read(Tokenizer* tokenizer) {
    auto cur = tokenizer->GetToken();
    if (tokenizer->IsEnd()) {
        try {
            throw SyntaxError("");
        } catch (const SyntaxError& e) {
            throw;
        }
    }
    if (std::holds_alternative<ConstantToken>(cur)) {
        int v = std::get<ConstantToken>(cur).value;
        tokenizer->Next();
        return std::make_shared<Number>(Number(v));
    } else if (std::holds_alternative<SymbolToken>(cur)) {
        std::string s = std::get<SymbolToken>(cur).name;
        tokenizer->Next();
        return std::make_shared<Symbol>(Symbol(s));
    } else if (std::holds_alternative<QuoteToken>(cur)) {
        tokenizer->Next();
        if (tokenizer->IsEnd()) {
            try {
                throw SyntaxError("");
            } catch (const SyntaxError& e) {
                throw;
            }
        }
        return std::make_shared<Cell>(
            Cell(std::make_shared<Symbol>(Symbol("quote")), ReadList(tokenizer, true)));
    } else if (std::holds_alternative<BracketToken>(cur)) {
        auto what = std::get<BracketToken>(cur);
        if (what == BracketToken::OPEN) {
            tokenizer->Next();
            return ReadList(tokenizer);
        } else {
            try {
                throw SyntaxError("");
            } catch (const SyntaxError& e) {
                throw;
            }
        }
    } else {
        try {
            throw SyntaxError("");
        } catch (const SyntaxError& e) {
            throw;
        }
    }
}
std::shared_ptr<Object> ReadList(Tokenizer* tokenizer, bool check) {
    if ((std::holds_alternative<BracketToken>(tokenizer->GetToken()) &&
         std::get<BracketToken>(tokenizer->GetToken()) == BracketToken::OPEN)) {
        auto first = Read(tokenizer);
        auto second = ReadList(tokenizer, check);
        return std::make_shared<Cell>(first, second);
    }
    if ((std::holds_alternative<QuoteToken>(tokenizer->GetToken()))) {
        tokenizer->Next();
        return std::make_shared<Cell>(
            Cell(std::make_shared<Symbol>(Symbol("quote")), ReadList(tokenizer, true)));
    }
    if (tokenizer->IsEnd()) {
        if (!check) {
            try {
                throw SyntaxError("");
            } catch (const SyntaxError& e) {
                throw;
            }
        } else {
            return nullptr;
        }
    }
    auto cur = tokenizer->GetToken();
    if (std::holds_alternative<BracketToken>(cur) &&
        std::get<BracketToken>(cur) == BracketToken::CLOSE) {
        tokenizer->Next();
        return nullptr;
    } else {
        auto first = Read(tokenizer);
        if (tokenizer->IsEnd()) {
            if (!check) {
                try {
                    throw SyntaxError("");
                } catch (const SyntaxError& e) {
                    throw;
                }
            } else {
                return std::make_shared<Cell>(first, nullptr);
            }
        }
        if (std::holds_alternative<DotToken>(tokenizer->GetToken())) {
            tokenizer->Next();
            auto second = Read(tokenizer);
            if ((!(std::holds_alternative<BracketToken>(tokenizer->GetToken()) &&
                   std::get<BracketToken>(tokenizer->GetToken()) == BracketToken::CLOSE))) {
                while (!tokenizer->IsEnd()) {
                    tokenizer->Next();
                }
                try {
                    throw SyntaxError("");
                } catch (const SyntaxError& e) {
                    throw;
                }
            }
            tokenizer->Next();
            return std::make_shared<Cell>(first, second);
        } else {
            auto second = ReadList(tokenizer, check);
            return std::make_shared<Cell>(first, second);
        }
    }
}