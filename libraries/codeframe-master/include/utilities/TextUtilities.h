#ifndef TEXTUTILITIES_H
#define TEXTUTILITIES_H

#include <cctype>
#include <algorithm>
#include <string>
#include <sstream>

namespace std {
    template<typename T>
    std::string to_string(const T &n) {
        std::ostringstream s;
        s << n;
        return s.str();
    }
}

namespace utilities
{
    namespace text
    {
        inline bool tobool(const std::string& x)
        {
            if(x == "1" || x == "true" || x == "TRUE") return true;
            else return false;
        }

        inline void split(const std::string& str, const std::string& delimiters , std::vector<std::string>& tokens)
        {
            // Skip delimiters at beginning.
            std::size_t lastPos = str.find_first_not_of(delimiters, 0);
            // Find first "non-delimiter".
            std::size_t pos     = str.find_first_of(delimiters, lastPos);

            while (std::string::npos != pos || std::string::npos != lastPos)
            {
                std::string foundString = str.substr(lastPos, pos - lastPos);
                // Found a token, add it to the vector.
                tokens.push_back(foundString);
                // Skip delimiters.  Note the "not_of"
                lastPos = str.find_first_not_of(delimiters, pos);
                // Find next "non-delimiter"
                pos = str.find_first_of(delimiters, lastPos);
            }
        };

        inline std::string stringtoupper( const std::string& in )
        {
            std::string out( in );
            std::transform(out.begin(), out.end(), out.begin(), (int(*)(int))std::toupper);
            return out;
        };

        const std::string WHITESPACE = " \n\r\t\f\v";

        inline std::string ltrim(const std::string& s)
        {
            size_t start = s.find_first_not_of(WHITESPACE);
            return (start == std::string::npos) ? "" : s.substr(start);
        }

        inline std::string rtrim(const std::string& s)
        {
            size_t end = s.find_last_not_of(WHITESPACE);
            return (end == std::string::npos) ? "" : s.substr(0, end + 1);
        }

        inline std::string trim(const std::string& s)
        {
            return rtrim(ltrim(s));
        }

        inline int stricmp(const std::string& str1, const std::string& str2)
        {
            return str1.compare( str2 );
        }

    }
}

#endif // TEXTUTILITIES_H
