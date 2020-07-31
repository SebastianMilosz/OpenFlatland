#include "MathUtilities.h"

#include <sstream>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string utilities::math::IntToStr(const uint32_t nbr)
{
    std::ostringstream s;
    s << nbr;
    std::string converted(s.str());

    return converted;
}
