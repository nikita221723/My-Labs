#pragma once

#include <string>
#include <parser.h>
#include <sstream>

class Interpreter {
public:
    Interpreter();
    std::shared_ptr<Object> ReadFull(const std::string& str);
    std::string Run(const std::string& s);
    std::string IWannaDie(const std::string& s);

private:
    std::unordered_map<std::string, std::shared_ptr<Object>> functions_;
};
