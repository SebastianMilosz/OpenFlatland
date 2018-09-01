#include "extpoint2d.hpp"

#include <vector>
#include <TextUtilities.h>
#include <MathUtilities.h>

namespace codeframe
{

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
void Point2D<int>::FromStringCallback ( StringType value )
{
    // Split using ; separator
    std::vector<std::string> pointPartsStrings;
    utilities::text::split(value, ";", pointPartsStrings);

    if( pointPartsStrings.size() == 2U )
    {
        m_x = utilities::math::StrToInt( pointPartsStrings[0] );
        m_y = utilities::math::StrToInt( pointPartsStrings[1] );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
void Point2D<int>::FromIntegerCallback( IntegerType value )
{
    m_x = value;
    m_y = value;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
void Point2D<int>::FromRealCallback( RealType value )
{
    m_x = value;
    m_y = value;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
StringType Point2D<int>::ToStringCallback() const
{
    std::string xString = utilities::math::IntToStr( m_x );
    std::string yString = utilities::math::IntToStr( m_y );

    std::string retVal = xString + std::string(";") + yString;

    return retVal;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
IntegerType Point2D<int>::ToIntegerCallback() const
{
    return 0;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
RealType Point2D<int>::ToRealCallback() const
{
    return 0.0F;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
Point2D<int>& Point2D<int>::operator=(const Point2D<int>& other)
{
    m_x = other.m_x;
    m_y = other.m_y;

    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
Point2D<int>& Point2D<int>::operator+(const Point2D<int>& rhs)
{
    m_x = m_x + rhs.m_x;
    m_y = m_y + rhs.m_y;

    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
bool Point2D<int>::operator==(const Point2D<int>& sval)
{
    if( (m_x == sval.m_x) && (m_y == sval.m_y) )
    {
        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
bool Point2D<int>::operator!=(const Point2D<int>& sval)
{
    if( (m_x != sval.m_x) || (m_y != sval.m_y) )
    {
        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void TemporaryFunction()
{
    Point2D<int> TempObj;
}

}
