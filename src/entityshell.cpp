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
EntityShell::EntityShell( std::string name, int x, int y, int z ) :
    PhysicsBody( name, NULL ),
    X   ( this, "X"   , 0 , cPropertyInfo().Kind( KIND_REAL ).Description("Xpos"), this, &EntityShell::GetX ),
    Y   ( this, "Y"   , 0 , cPropertyInfo().Kind( KIND_REAL ).Description("Ypos"), this, &EntityShell::GetY ),
    Z   ( this, "Z"   , 0 , cPropertyInfo().Kind( KIND_REAL ).Description("Zpos"), this, &EntityShell::GetZ ),
    Name( this, "Name", "", cPropertyInfo().Kind( KIND_TEXT ).Description("Name") )
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
    X   ( this, "X"   , 0, cPropertyInfo().Kind( KIND_REAL ).Description("Xpos"), this, &EntityShell::GetX ),
    Y   ( this, "Y"   , 0, cPropertyInfo().Kind( KIND_REAL ).Description("Ypos"), this, &EntityShell::GetY ),
    Z   ( this, "Z"   , 0, cPropertyInfo().Kind( KIND_REAL ).Description("Zpos"), this, &EntityShell::GetZ ),
    Name( this, "Name", 0, cPropertyInfo().Kind( KIND_TEXT ).Description("Name") )
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
unsigned int EntityShell::GetX()
{
    if( GetDescriptor().Body == NULL ) return 0;

    return GetDescriptor().Body->GetPosition().x * sDescriptor::PIXELS_IN_METER;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::SetX(unsigned int val)
{
    //m_x = val;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
unsigned int EntityShell::GetY()
{
    if( GetDescriptor().Body == NULL ) return 0;

    return GetDescriptor().Body->GetPosition().y * sDescriptor::PIXELS_IN_METER;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::SetY(unsigned int val)
{
    //m_y = val;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
unsigned int EntityShell::GetZ()
{
    return 0;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::SetZ(unsigned int val)
{
    //m_z = val;
}
