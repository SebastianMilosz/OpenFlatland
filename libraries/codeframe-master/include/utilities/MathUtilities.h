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

namespace utilities
{
	namespace math
	{
		/*****************************************************************************/
		/**
		  * @brief
		 **
		******************************************************************************/
		inline std::string IntToStr(int nbr)
		{
		    if(nbr > 999999999) return std::string("GetStringFromInt - Overload");
		    char buffer [12];
		    sprintf (buffer, "%d",nbr);
		    return std::string(buffer);
        }

        /*****************************************************************************/
        /**
          * @brief
         **
        ******************************************************************************/
        inline std::string IntToHex(unsigned int nbr, std::string prefix = "", std::string separator = ":", char trim = 0)
        {
            unsigned char* buf = (unsigned char*)(&nbr);

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
         **
        ******************************************************************************/
        inline std::string LongToHex(long nbr)
        {
            return IntToHex( (unsigned int)nbr );
        }

		/*****************************************************************************/
		/**
		  * @brief
		 **
		******************************************************************************/
		inline std::string FloatToStr(float nbr)
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
		inline unsigned int HexToInt(std::string hex)
		{
            return strtol(hex.c_str(), NULL, 16);
		}

        /*****************************************************************************/
        /**
          * @brief
         **
        ******************************************************************************/
        inline int StrToInt( std::string strnbr )
        {
            return strtol(strnbr.c_str(), NULL, 10);
        }

    }
}

#endif
