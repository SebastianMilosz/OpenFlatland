#include "MathUtilities.h"

#include <sstream>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string utilities::math::IntToStr(int nbr)
{
    std::ostringstream s;
    s << nbr;
    std::string converted(s.str());

    return converted;
}
