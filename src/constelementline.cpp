#include "constelementline.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElementLine::ConstElementLine( const std::string& name, codeframe::Point2D<int>& startPoint, codeframe::Point2D<int>& endPoint ) :
    ConstElement( name ),
    StartPoint( this, "SPoint" , Point2D<int>( startPoint ), cPropertyInfo().Kind( KIND_2DPOINT ).Description("StartPoint"), this ),
    EndPoint  ( this, "EPoint" , Point2D<int>( endPoint   ), cPropertyInfo().Kind( KIND_2DPOINT ).Description("EndPoint"), this)
{
    float sx( (float)startPoint.X()/sDescriptor::PIXELS_IN_METER );
    float sy( (float)startPoint.Y()/sDescriptor::PIXELS_IN_METER );
    float w( endPoint.X() / sDescriptor::PIXELS_IN_METER - sx );
    float h( endPoint.Y() / sDescriptor::PIXELS_IN_METER - sy );

    b2EdgeShape * lineShape = new b2EdgeShape ();
    lineShape->Set(
                   b2Vec2( 0, 0 ),
                   b2Vec2( w, h )
                  );

    GetDescriptor().Shape = lineShape;
    GetDescriptor().BodyDef.position = b2Vec2( sx, sy );
    GetDescriptor().BodyDef.type = b2_staticBody;
    GetDescriptor().BodyDef.userData = (void*)this;
    GetDescriptor().FixtureDef.density = 1.f;
    GetDescriptor().FixtureDef.friction = 0.f;
    GetDescriptor().FixtureDef.restitution = 1.f;
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
    StartPoint( this, "SPoint" , Point2D<int>(), cPropertyInfo().Kind( KIND_2DPOINT ).Description("StartPoint"), this ),
    EndPoint  ( this, "EPoint" , Point2D<int>(), cPropertyInfo().Kind( KIND_2DPOINT ).Description("EndPoint"), this)
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
    if( (b2Body*)nullptr != body )
    {
        // @todo Support line color
        //sf::Color& entColor = GetColor();

        float xpos( sDescriptor::PIXELS_IN_METER * body->GetPosition().x );
        float ypos( sDescriptor::PIXELS_IN_METER * body->GetPosition().y );

        float sx( (float)StartPoint.GetValue().X() );
        float sy( (float)StartPoint.GetValue().Y() );
        float w( EndPoint.GetValue().X() - sx );
        float h( EndPoint.GetValue().Y() - sy );

        sf::Vertex line[2];
        line[0].position = sf::Vector2f(xpos, ypos);
        line[0].color  = sf::Color::Red;
        line[1].position = sf::Vector2f(xpos+w, ypos+h);
        line[1].color = sf::Color::Red;

        window.draw( line, 2, sf::Lines );
    }
}
