#include "entityshell.h"

static const float SCALE = 30.f;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell::EntityShell( b2World& world, int x, int y, int z )
{
    b2BodyDef BodyDef;
    BodyDef.position = b2Vec2(x/SCALE, y/SCALE);
    BodyDef.type = b2_dynamicBody;
    m_Body = world.CreateBody(&BodyDef);

    b2PolygonShape Shape;
    Shape.SetAsBox((32.f/2)/SCALE, (32.f/2)/SCALE);
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

    return m_Body->GetPosition().x;
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

    return m_Body->GetPosition().y;
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
