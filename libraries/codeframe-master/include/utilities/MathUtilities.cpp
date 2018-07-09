#include "MathUtilities.h"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string utilities::math::IntToStr(int nbr)
{
    if ( nbr > 999999999 )
    {
        return std::string("GetStringFromInt - Overload");
    }
    char buffer [12];
    sprintf (buffer, "%d",nbr);
    return std::string(buffer);
}
