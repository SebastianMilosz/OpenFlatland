#include "utilities/FileUtilities.h"
#include "utilities/MathUtilities.h"

#include <algorithm>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
struct MatchPathSeparator
{
    bool operator()( char ch ) const
    {
        return ch == '/';
    }
};

/*****************************************************************************/
/**
  * @brief Zwraca nazwe pliku
 **
******************************************************************************/
std::string utilities::file::GetFileName( std::string const& pathname )
{
    return std::string( std::find_if( pathname.rbegin(), pathname.rend(), MatchPathSeparator() ).base(), pathname.end() );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
utilities::url::cIP::cIP(void)
{
    addr[0] = 0;
    addr[1] = 0;
    addr[2] = 0;
    addr[3] = 0;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string utilities::url::cIP::ToString()
{
    return utilities::math::IntToStr( addr[0] ) + std::string(".") +
           utilities::math::IntToStr( addr[1] ) + std::string(".") +
           utilities::math::IntToStr( addr[2] ) + std::string(".") +
           utilities::math::IntToStr( addr[3] );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
utilities::url::cIP& utilities::url::cIP::FromString( std::string str )
{
    if( str == "localhost" ) return *this;

    std::stringstream s( str );
    int a,b,c,d; //to store the 4 ints
    char ch; //to temporarily store the '.'
    s >> a >> ch >> b >> ch >> c >> ch >> d;

    addr[0] = a;
    addr[1] = b;
    addr[2] = c;
    addr[3] = d;

    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
utilities::url::cIP& utilities::url::cIP::FromInteger( uint8_t v1, uint8_t v2, uint8_t v3, uint8_t v4 )
{
    addr[0] = v1;
    addr[1] = v2;
    addr[2] = v3;
    addr[3] = v4;

    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
uint32_t utilities::url::cIP::ToIntAdr() const
{
    return *((uint32_t*)addr);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
uint8_t* utilities::url::cIP::ToIntAdrPtr()
{
    return addr;
}
