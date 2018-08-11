#include "constelementline.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElementLine::ConstElementLine( std::string name, codeframe::Point2D& startPoint, codeframe::Point2D& endPoint ) :
    ConstElement( name ),
    StartPoint( this, "StartPoint" , Point2D(), cPropertyInfo().Kind( KIND_2DPOINT ).Description("StartPoint"), this ),
    EndPoint  ( this, "EndPoint"   , Point2D(), cPropertyInfo().Kind( KIND_2DPOINT ).Description("EndPoint"), this)
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElementLine::~ConstElementLine()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElementLine::ConstElementLine(const ConstElementLine& other) :
    ConstElement( other ),
    StartPoint( this, "SPoint" , Point2D(), cPropertyInfo().Kind( KIND_2DPOINT ).Description("StartPoint"), this ),
    EndPoint  ( this, "EPoint" , Point2D(), cPropertyInfo().Kind( KIND_2DPOINT ).Description("EndPoint"), this)
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElementLine& ConstElementLine::operator=(const ConstElementLine& other)
{
    return *this;
}
