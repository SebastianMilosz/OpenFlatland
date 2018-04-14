#include "entityshell.h"

static const float PIXELS_IN_METER = 30.f;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell::EntityShell( World& world, int x, int y, int z ) :
    m_color(sf::Color::Red)
{
    b2BodyDef BodyDef;
    BodyDef.position = b2Vec2((float)x/PIXELS_IN_METER, (float)y/PIXELS_IN_METER);
    BodyDef.type = b2_dynamicBody;
    BodyDef.userData = (void*)this;
    m_Body = world.CreateBody(&BodyDef);

    b2PolygonShape Shape;
    Shape.SetAsBox((32.f/2)/PIXELS_IN_METER, (32.f/2)/PIXELS_IN_METER);

    b2FixtureDef FixtureDef;
    FixtureDef.density = 1.f;
    FixtureDef.friction = 0.7f;
    FixtureDef.shape = &Shape;

    m_Body->CreateFixture(&FixtureDef);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell::~EntityShell()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell::EntityShell(const EntityShell& other)
{
    //copy ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell& EntityShell::operator=(const EntityShell& rhs)
{
    if (this == &rhs) return *this; // handle self assignment
    //assignment operator
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
unsigned int EntityShell::GetX()
{
    if( m_Body == NULL ) return 0;

    return m_Body->GetPosition().x * PIXELS_IN_METER;
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
    if( m_Body == NULL ) return 0;

    return m_Body->GetPosition().y * PIXELS_IN_METER;
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
