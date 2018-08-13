#include "extendedtypepoint2d.hpp"

#include <vector>
#include <TextUtilities.h>
#include <MathUtilities.h>

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Point2D::Point2D() :
    m_x( 0 ),
    m_y( 0 )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Point2D::Point2D( int x, int y ) :
    m_x( x ),
    m_y( y )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Point2D::Point2D( const Point2D& other ) :
    m_x( other.m_x ),
    m_y( other.m_y )
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Point2D::~Point2D()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void Point2D::FromStringCallback ( StringType value )
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
void Point2D::FromIntegerCallback( IntegerType value )
{
    m_x = value;
    m_y = value;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void Point2D::FromRealCallback( RealType value )
{
    m_x = value;
    m_y = value;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
StringType Point2D::ToStringCallback() const
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
IntegerType Point2D::ToIntegerCallback() const
{
    return 0;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
RealType Point2D::ToRealCallback() const
{
    return 0.0F;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Point2D& Point2D::operator=(const Point2D& other)
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
Point2D& Point2D::operator+(const Point2D& rhs)
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
bool Point2D::operator==(const Point2D& sval)
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
bool Point2D::operator!=(const Point2D& sval)
{
    if( (m_x != sval.m_x) || (m_y != sval.m_y) )
    {
        return true;
    }
    return false;
}
