#include "entityshell.hpp"

#include <utilities/LoggerUtilities.h>
#include <utilities/TextUtilities.h>

#include "fontfactory.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell::EntityShell( std::string name, int x, int y ) :
    PhysicsBody( name, NULL ),
    X       ( this, "X"       , 0    , cPropertyInfo().Kind( KIND_NUMBER ).Description("Xpos"), this, &EntityShell::GetX ),
    Y       ( this, "Y"       , 0    , cPropertyInfo().Kind( KIND_NUMBER ).Description("Ypos"), this, &EntityShell::GetY ),
    CastRays( this, "CastRays", false, cPropertyInfo().Kind( KIND_LOGIC  ).Description("CastRays") ),
    RaysCnt ( this, "RaysCnt" , 100U , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysCnt") ),
    Name    ( this, "Name"    , ""   , cPropertyInfo().Kind( KIND_TEXT   ).Description("Name") ),
    Density ( this, "Density" , 1.F  , cPropertyInfo().Kind( KIND_REAL   ).Description("Density") ),
    Friction( this, "Friction", 0.7F , cPropertyInfo().Kind( KIND_REAL   ).Description("Friction") )
{
    b2CircleShape* shape =  new b2CircleShape();
    shape->m_p.Set(0, 0);
    shape->m_radius = 15.0f/sDescriptor::PIXELS_IN_METER;

    GetDescriptor().Shape = shape;
    GetDescriptor().BodyDef.position = b2Vec2(
                                              (float)x/sDescriptor::PIXELS_IN_METER,
                                              (float)y/sDescriptor::PIXELS_IN_METER
                                             );
    GetDescriptor().BodyDef.type = b2_dynamicBody;
    GetDescriptor().BodyDef.userData    = (void*)this;
    GetDescriptor().FixtureDef.density  = (float)Density;
    GetDescriptor().FixtureDef.friction = (float)Friction;
    GetDescriptor().FixtureDef.shape    = GetDescriptor().Shape;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell::~EntityShell()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell::EntityShell(const EntityShell& other) :
    PhysicsBody( other ),
    X       ( other.X ),
    Y       ( other.Y ),
    CastRays( other.CastRays ),
    RaysCnt ( other.RaysCnt ),
    Name    ( other.Name ),
    Density ( other.Density ),
    Friction( other.Friction )
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell& EntityShell::operator=(const EntityShell& rhs)
{
    PhysicsBody::operator = (rhs);

    //assignment operator
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::Draw( sf::RenderWindow& window, b2Body* body )
{
    if( (b2Body*)NULL != body )
    {
        sf::Color& entColor = GetColor();

        float xpos = body->GetPosition().x;
        float ypos = body->GetPosition().y;

        sf::CircleShape circle;
        circle.setRadius(sDescriptor::PIXELS_IN_METER * 0.5f);
        circle.setOutlineColor( entColor );

        if( IsSelected() == true )
        {
            circle.setFillColor( sf::Color::Blue );
        }
        else
        {
            circle.setFillColor( sf::Color::Black );
        }

        circle.setOutlineThickness(3);
        circle.setOrigin(16.f, 16.f);
        circle.setPosition(sDescriptor::PIXELS_IN_METER * xpos, sDescriptor::PIXELS_IN_METER * ypos);
        circle.setRotation(body->GetAngle() * 180/b2_pi);
        window.draw(circle);

/*
        sf::Text text;
        text.setString( std::string("(") + std::to_string(xpos) + std::string(", ") + std::to_string(ypos) + std::string(")") );
        text.setColor(sf::Color::White);
        text.setCharacterSize(12);
        text.setFont( FontFactory::GetFont() );
        text.setPosition(sDescriptor::PIXELS_IN_METER * xpos, sDescriptor::PIXELS_IN_METER * ypos);
        window.draw(text);
*/
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int EntityShell::GetX()
{
    if( GetDescriptor().Body == NULL ) return 0;

    return GetDescriptor().Body->GetPosition().x * sDescriptor::PIXELS_IN_METER;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::SetX(int val)
{
    //m_x = val;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int EntityShell::GetY()
{
    if( GetDescriptor().Body == NULL ) return 0;

    return GetDescriptor().Body->GetPosition().y * sDescriptor::PIXELS_IN_METER;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::SetY(int val)
{
    //m_y = val;
}
