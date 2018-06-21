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
    cSerializable( name, NULL ),
    X( this, "X", &EntityShell::GetX, cPropertyInfo().Kind( KIND_REAL ).Description("Xpos") ),
    Y( this, "Y", &EntityShell::GetY, cPropertyInfo().Kind( KIND_REAL ).Description("Ypos") ),
    Z( this, "Z", &EntityShell::GetZ, cPropertyInfo().Kind( KIND_REAL ).Description("Zpos") )
{
    m_descryptor.Body = NULL;

    m_descryptor.Shape.m_p.Set(0, 0);
    m_descryptor.Shape.m_radius = 15.0f/PIXELS_IN_METER;
    m_descryptor.BodyDef.position = b2Vec2((float)x/PIXELS_IN_METER, (float)y/PIXELS_IN_METER);
    m_descryptor.BodyDef.type = b2_dynamicBody;
    m_descryptor.BodyDef.userData = (void*)this;
    m_descryptor.FixtureDef.density = 1.f;
    m_descryptor.FixtureDef.friction = 0.7f;
    m_descryptor.FixtureDef.shape = &m_descryptor.Shape;
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
EntityShell::sEntityShellDescriptor& EntityShell::GetDescriptor()
{
    return m_descryptor;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell::EntityShell(const EntityShell& other) :
    cSerializable( other ),
    X( this, "X", &EntityShell::GetX, cPropertyInfo().Kind( KIND_REAL ).Description("Xpos") ),
    Y( this, "Y", &EntityShell::GetY, cPropertyInfo().Kind( KIND_REAL ).Description("Ypos") ),
    Z( this, "Z", &EntityShell::GetZ, cPropertyInfo().Kind( KIND_REAL ).Description("Zpos") )
{
    m_descryptor.Body = other.m_descryptor.Body;
    m_descryptor.Shape = other.m_descryptor.Shape;
    m_descryptor.FixtureDef = other.m_descryptor.FixtureDef;
    m_descryptor.BodyDef = other.m_descryptor.BodyDef;
    m_descryptor.Color = other.m_descryptor.Color;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell& EntityShell::operator=(const EntityShell& rhs)
{
    if ( this == &rhs )
    {
        m_descryptor.Body = rhs.m_descryptor.Body;
        m_descryptor.Shape = rhs.m_descryptor.Shape;
        m_descryptor.FixtureDef = rhs.m_descryptor.FixtureDef;
        m_descryptor.BodyDef = rhs.m_descryptor.BodyDef;
        m_descryptor.Color = rhs.m_descryptor.Color;
    }
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
    if( m_descryptor.Body == NULL ) return 0;

    return m_descryptor.Body->GetPosition().x * PIXELS_IN_METER;
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
    if( m_descryptor.Body == NULL ) return 0;

    return m_descryptor.Body->GetPosition().y * PIXELS_IN_METER;
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
