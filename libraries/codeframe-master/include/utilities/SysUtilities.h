#ifndef SYSUTILITIES_H
#define SYSUTILITIES_H

#include <string>
#include <sstream>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <sys/timeb.h>

namespace utilities
{
    namespace system
    {
        /*****************************************************************************/
        /**
          * @brief Zwraca aktualny czas
         **
        ******************************************************************************/
        inline std::string GetNow()
        {
            time_t rawtime;
            time ( &rawtime );
            std::string retDateTime(ctime(&rawtime));

            unsigned int pos;
            if((pos = retDateTime.find('\n')) != std::string::npos)
                    retDateTime.erase(pos);

            return retDateTime;
        }

        std::string GetNowPrecise();

        /*****************************************************************************/
        /**
          * @brief
         **
        ******************************************************************************/
        inline double GetTime( void )
        {
            struct timeb timeStamp;
            ftime( &timeStamp );
            double nTime = (double) timeStamp.time*1000 + timeStamp.millitm;
            return nTime;
        }

        /*****************************************************************************/
        /**
          * @brief
         **
        ******************************************************************************/
        inline clock_t GetTimeMs( void )
        {
            return clock();
        }
    }
}

#endif // SYSUTILITIES_H
