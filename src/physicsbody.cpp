#include "physicsbody.h"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
PhysicsBody::PhysicsBody( std::string name, codeframe::cSerializableInterface* parent ) :
    cSerializable( name, parent )
{
    m_descryptor.Body = NULL;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
PhysicsBody::~PhysicsBody()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
PhysicsBody::PhysicsBody(const PhysicsBody& other) :
    cSerializable( other )
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
PhysicsBody& PhysicsBody::operator=(const PhysicsBody& rhs)
{
    if ( this == &rhs )
    {
        m_descryptor.Body = rhs.m_descryptor.Body;
        m_descryptor.Shape = rhs.m_descryptor.Shape;
        m_descryptor.FixtureDef = rhs.m_descryptor.FixtureDef;
        m_descryptor.BodyDef = rhs.m_descryptor.BodyDef;
        m_descryptor.Color = rhs.m_descryptor.Color;
    }
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PhysicsBody::SetColor( const sf::Color& color )
{
    m_descryptor.Color = color;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
sf::Color& PhysicsBody::GetColor()
{
    return m_descryptor.Color;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
PhysicsBody::sEntityShellDescriptor& PhysicsBody::GetDescriptor()
{
    return m_descryptor;
}