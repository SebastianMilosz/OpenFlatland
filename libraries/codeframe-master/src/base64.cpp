#include "base64.hpp"

#include <ctype.h>

namespace codeframe
{

static const char* base64_charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
static const std::string base64_chars = base64_charset;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
static inline bool is_base64(unsigned char c)
{
    return (isalnum(c) || (c == '+') || (c == '/'));
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string base64_encode( const uint8_t *indata, size_t size )
{
    std::string outdata;
    outdata.reserve(((size * 8) / 6) + 2);
    std::string::size_type remaining = size;
    const char* dp = (const char*)indata;

    while ( remaining >= 3 )
    {
        outdata.push_back( base64_charset[(dp[0] & 0xfc) >> 2] );
        outdata.push_back( base64_charset[((dp[0] & 0x03) << 4) | ((dp[1] & 0xf0) >> 4)] );
        outdata.push_back( base64_charset[((dp[1] & 0x0f) << 2) | ((dp[2] & 0xc0) >> 6)] );
        outdata.push_back( base64_charset[(dp[2] & 0x3f)] );
        remaining -= 3;
        dp += 3;
    }

    if ( remaining == 2 )
    {
        outdata.push_back( base64_charset[(dp[0] & 0xfc) >> 2] );
        outdata.push_back( base64_charset[((dp[0] & 0x03) << 4) | ((dp[1] & 0xf0) >> 4)] );
        outdata.push_back( base64_charset[((dp[1] & 0x0f) << 2)] );
        outdata.push_back( base64_charset[64] );
    }
    else if ( remaining == 1 )
    {
        outdata.push_back( base64_charset[(dp[0] & 0xfc) >> 2] );
        outdata.push_back( base64_charset[((dp[0] & 0x03) << 4)] );
        outdata.push_back( base64_charset[64] );
        outdata.push_back( base64_charset[64] );
    }

    return outdata;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string base64_decode( const std::string& encoded_string )
{
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4];
    unsigned char char_array_3[3];
    std::string ret;

    while ( in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_]) )
    {
        char_array_4[i++] = encoded_string[in_];
        in_++;

        if ( i == 4 )
        {
            for ( i = 0; i < 4; i++ )
            {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for ( i = 0; (i < 3); i++ )
            {
                ret += char_array_3[i];
            }
            i = 0;
        }
    }

    if ( i )
    {
        for ( j = i; j < 4; j++ )
        {
            char_array_4[j] = 0;
        }

        for ( j = 0; j < 4; j++ )
        {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for ( j = 0; (j < i - 1); j++ )
        {
            ret += char_array_3[j];
        }
    }

    return ret;
}

}
