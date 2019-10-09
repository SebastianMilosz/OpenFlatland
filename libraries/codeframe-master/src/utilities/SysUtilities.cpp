#include "utilities/SysUtilities.h"

#include <stdint.h>
#include <sys/timeb.h>
#include <sys/types.h>  // pid_t

#ifdef _WIN32
#include <windows.h>
#endif // WIN32

/*****************************************************************************/
/**
 * @brief   Return current date and time in the same format like GetNow() with
 *          miliseconds
 * @return  Date and time std::string
 **
******************************************************************************/
std::string utilities::system::GetNowPrecise()
{
    #ifdef _WIN32
        std::string retDateTime = GetNow();

        SYSTEMTIME st;
        GetSystemTime(&st);

        std::stringstream milisecondsString;
        milisecondsString << ".";
        milisecondsString.fill('0');
        milisecondsString.width(2);
        milisecondsString << st.wMilliseconds;

        uint8_t secPos = retDateTime.find_last_of(' ');
        retDateTime.insert(secPos, milisecondsString.str());

        return retDateTime;
    #else // _WIN32
        #error "NOT SUPPORTED PLATFORM"
    #endif
}
