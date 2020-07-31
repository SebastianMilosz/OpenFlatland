#ifndef mathUtilitiesH
#define mathUtilitiesH

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <ios>
#include <iomanip>
#include <stdint.h>
#include <stdlib.h>
#include <limits>
#include <cmath>

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
    namespace math
    {
        typedef float float32;

        std::string IntToStr( const uint32_t nbr );

        /*****************************************************************************/
        /**
          * @brief
         **
        ******************************************************************************/
        inline std::string IntToHex(const uint32_t nbr, const std::string& prefix = "", const std::string& separator = ":", char trim = 0)
        {
            const uint8_t* buf = (const uint8_t*)(&nbr);

            std::ostringstream s;
            s << std::hex << std::setfill('0') << std::uppercase << prefix;

            // Trim off
            if( trim == 0 )
            {
                s << std::setw(2) << static_cast<int>(buf[3]) << separator;
                s << std::setw(2) << static_cast<int>(buf[2]) << separator;
                s << std::setw(2) << static_cast<int>(buf[1]) << separator;
                s << std::setw(2) << static_cast<int>(buf[0]);
            }
            // Trim auto
            else if( trim == -1 )
            {
                if(buf[3]) { s << std::setw(2) << static_cast<int>(buf[3]) << separator; }
                if(buf[2]) { s << std::setw(2) << static_cast<int>(buf[2]) << separator; }
                if(buf[1]) { s << std::setw(2) << static_cast<int>(buf[1]) << separator; }

                s << std::setw(2) << static_cast<int>(buf[0]);
            }
            // Trim 1
            else if( trim == 1 )
            {
                s << std::setw(2) << static_cast<int>(buf[0]);
            }
            // Trim 2
            else if( trim == 2 )
            {
                s << std::setw(2) << static_cast<int>(buf[1]) << separator;
                s << std::setw(2) << static_cast<int>(buf[0]);
            }
            // Trim 3
            else if( trim == 3 )
            {
                s << std::setw(2) << static_cast<int>(buf[2]) << separator;
                s << std::setw(2) << static_cast<int>(buf[1]) << separator;
                s << std::setw(2) << static_cast<int>(buf[0]);
            }
            // Trim 4
            else if( trim == 4 )
            {
                s << std::setw(2) << static_cast<int>(buf[3]) << separator;
                s << std::setw(2) << static_cast<int>(buf[2]) << separator;
                s << std::setw(2) << static_cast<int>(buf[1]) << separator;
                s << std::setw(2) << static_cast<int>(buf[0]);
            }

            std::string retString = s.str();

            return retString;
        }

        /*****************************************************************************/
        /**
          * @brief
          * @todo fix this temporary code someday
         **
        ******************************************************************************/
        inline std::string PointerToHex(void* ptr)
        {
            return IntToHex( *((uint32_t*)&ptr) );
        }

        /*****************************************************************************/
        /**
          * @brief
          * @todo fix this temporary code someday
         **
        ******************************************************************************/
        inline std::string LongToHex( long nbr )
        {
            return IntToHex( (uint32_t)nbr );
        }

        /*****************************************************************************/
        /**
        * @brief
        **
        ******************************************************************************/
        inline std::string FloatToStr( float nbr )
        {
            if(nbr > 999999999) return std::string("GetStringFromInt - Overload");
            char buffer [12];
            sprintf (buffer, "%f",nbr);
            return std::string(buffer);
        }

        /*****************************************************************************/
        /**
        * @brief
        **
        ******************************************************************************/
        inline std::string DoubleToStr( double nbr )
        {
            std::ostringstream ss;

            ss << nbr;

            return std::string( ss.str() );
        }

        /*****************************************************************************/
        /**
        * @brief
        **
        ******************************************************************************/
        inline unsigned int HexToInt( const std::string& hex)
        {
            return strtol(hex.c_str(), NULL, 16);
        }

        /*****************************************************************************/
        /**
          * @brief
         **
        ******************************************************************************/
        inline int StrToInt( const std::string& strnbr )
        {
            return strtol(strnbr.c_str(), NULL, 10);
        }

        /*****************************************************************************/
        /**
          * @brief
         **
        ******************************************************************************/
        inline float StrToFloat( const std::string& strnbr )
        {
            return atof( strnbr.c_str() );
        }

        /*****************************************************************************/
        /**
          * @brief
         **
        ******************************************************************************/
        inline double StrToDouble( const std::string& strnbr )
        {
            return atof( strnbr.c_str() );
        }

        /*****************************************************************************/
        /**
          * @brief
         **
        ******************************************************************************/
        inline float32 ConstrainAngle( float32 x )
        {
            x = fmod( x, 360 );
            if ( x < 0 )
            {
                x += 360;
            }
            return x;
        }

    }
}

#endif
