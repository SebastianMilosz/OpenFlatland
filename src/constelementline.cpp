#include "constelementline.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElementLine::ConstElementLine( std::string name, codeframe::Point2D& startPoint, codeframe::Point2D& endPoint ) :
    ConstElement( name ),
    StartPoint( this, "SPoint" , Point2D( startPoint ), cPropertyInfo().Kind( KIND_2DPOINT ).Description("StartPoint"), this ),
    EndPoint  ( this, "EPoint"   , Point2D( endPoint   ), cPropertyInfo().Kind( KIND_2DPOINT ).Description("EndPoint"), this)
{
    b2EdgeShape * lineShape = new b2EdgeShape ();
    lineShape->Set(
                   b2Vec2( startPoint.X() / sDescriptor::PIXELS_IN_METER, startPoint.Y() / sDescriptor::PIXELS_IN_METER ),
                   b2Vec2( endPoint.X()   / sDescriptor::PIXELS_IN_METER, endPoint.Y()   / sDescriptor::PIXELS_IN_METER )
                  );

    GetDescriptor().Shape = lineShape;
    GetDescriptor().BodyDef.position = b2Vec2(
                                              (float)startPoint.X()/sDescriptor::PIXELS_IN_METER,
                                              (float)startPoint.Y()/sDescriptor::PIXELS_IN_METER
                                             );
    GetDescriptor().BodyDef.type = b2_staticBody;
    GetDescriptor().BodyDef.userData = (void*)this;
    GetDescriptor().FixtureDef.density = 1.f;
    GetDescriptor().FixtureDef.friction = 0.7f;
    GetDescriptor().FixtureDef.shape = GetDescriptor().Shape;
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

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ConstElementLine::Draw( sf::RenderWindow& window, b2Body* body )
{

}
