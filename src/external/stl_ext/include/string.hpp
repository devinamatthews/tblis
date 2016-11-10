#ifndef _STL_EXT_STRING_HPP_
#define _STL_EXT_STRING_HPP_

#include <cctype>
#include <string>
#include <sstream>
#include <utility>

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
        for (int i = 0;i < s.size();i++) trans[(unsigned char)s[i]] = s[i];
    }
    else
    {
        for (int i = 0;i < 256;i++) trans[i] = i;
    }

    for (int i = 0;i < n;i++) trans[(unsigned char)from[i]] = to[i];
    for (int i = 0;i < s.size();i++) s[i] = trans[(unsigned char)s[i]];

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
    string s(move(S));
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

}

#endif
