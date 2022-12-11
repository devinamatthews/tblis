#ifndef _STL_EXT_STRING_HPP_
#define _STL_EXT_STRING_HPP_

#include <cctype>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include <iostream>

namespace stl_ext
{

using std::string;

template <typename T> string str(const T& t)
{
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

inline string& str(string& str)
{
    return str;
}

inline const string& str(const string& str)
{
    return str;
}

inline string&& str(string&& str)
{
    return std::move(str);
}

inline const char* str(const char* str)
{
    return str;
}

template <typename T, typename... U>
string str(const string& fmt, T&& t, U&&... u)
{
    std::ostringstream oss;
    oss << printos(fmt, std::forward<T>(t), std::forward<U>(u)...);
    return oss.str();
}

inline string& translate(string& s, const string& from, const string& to)
{
    size_t n = std::min(from.size(), to.size());

    unsigned char trans[256];
    if (s.size() < 256)
    {
        for (size_t i = 0;i < s.size();i++) trans[(unsigned char)s[i]] = s[i];
    }
    else
    {
        for (int i = 0;i < 256;i++) trans[i] = i;
    }

    for (size_t i = 0;i < n;i++) trans[(unsigned char)from[i]] = to[i];
    for (size_t i = 0;i < s.size();i++) s[i] = trans[(unsigned char)s[i]];

    return s;
}

inline string translated(string s, const string& from, const string& to)
{
    translate(s, from, to);
    return s;
}

inline string toupper(string&& s)
{
    string S(std::move(s));
    for (auto& C : S) C = std::toupper(C);
    return S;
}

inline string tolower(string&& S)
{
    string s(std::move(S));
    for (auto& c : s) c = std::tolower(c);
    return s;
}

inline string toupper(const string& s)
{
    string S(s);
    for (auto& C : S) C = std::toupper(C);
    return S;
}

inline string tolower(const string& S)
{
    string s(S);
    for (auto& c : s) c = std::tolower(c);
    return s;
}

inline std::string trim(const std::string& s)
{
    auto begin = s.find_first_not_of(" \n\r\t");
    auto end = s.find_last_not_of(" \n\r\t");

    if (begin == s.npos) return "";
    else return s.substr(begin, end-begin+1);
}

inline std::vector<std::string> split(const std::string& s,
                                      const std::string& sep = "",
                                      int max_split = -1)
{
    std::vector<std::string> tokens;

    if (sep == "")
    {
        std::istringstream iss(s);
        std::string token;
        for (auto i = 0;(i < max_split || max_split == -1) && (iss >> token);i++)
            tokens.push_back(token);

        token.clear();
        char c;
        while (iss.get(c)) token.push_back(c);
        if (!token.empty()) tokens.push_back(token);
    }
    else
    {
        auto begin = 0;
        for (auto i = 0;i < max_split || max_split == -1;i++)
        {
            auto end = s.find(sep, begin);

            if (end == s.npos)
            {
                tokens.push_back(s.substr(begin));
                begin = end;
                break;
            }
            else
            {
                tokens.push_back(s.substr(begin, end-begin));
                begin = end+sep.size();
            }
        }

        if (begin != s.npos)
            tokens.push_back(s.substr(begin));
    }

    return tokens;
}

}

#endif
