#ifndef TEXTUTILITIES_H
#define TEXTUTILITIES_H

#include <cctype>
#include <algorithm>

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

        inline std::string stringtoupper( std::string in )
        {
            std::string out( in );
            std::transform(out.begin(), out.end(), out.begin(), (int(*)(int))std::toupper);
            return out;
        };
	}
}

#endif // TEXTUTILITIES_H
