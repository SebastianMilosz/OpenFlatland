#include "utilities/SysUtilities.h"

#include <iostream>

/*****************************************************************************/
/**
 * @brief   Return current date and time in the same format like GetNow() with
 *          miliseconds
 * @return  Date and time std::string
 **
******************************************************************************/
std::string utilities::system::GetNowPrecise()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);

    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}
