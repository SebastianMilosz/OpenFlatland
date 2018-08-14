#include "entityshell.h"

#include <utilities/LoggerUtilities.h>

using namespace codeframe;

static const float PIXELS_IN_METER = 30.f;

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
    GetDescriptor().Shape.m_p.Set(0, 0);
    GetDescriptor().Shape.m_radius = 15.0f/PIXELS_IN_METER;
    GetDescriptor().BodyDef.position = b2Vec2((float)x/PIXELS_IN_METER, (float)y/PIXELS_IN_METER);
    GetDescriptor().BodyDef.type = b2_dynamicBody;
    GetDescriptor().BodyDef.userData = (void*)this;
    GetDescriptor().FixtureDef.density = 1.f;
    GetDescriptor().FixtureDef.friction = 0.7f;
    GetDescriptor().FixtureDef.shape = &GetDescriptor().Shape;
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
unsigned int EntityShell::GetX()
{
    if( GetDescriptor().Body == NULL ) return 0;

    return GetDescriptor().Body->GetPosition().x * PIXELS_IN_METER;
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

    return GetDescriptor().Body->GetPosition().y * PIXELS_IN_METER;
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
