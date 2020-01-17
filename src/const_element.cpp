#include "const_element.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElement::ConstElement( const std::string& name ) :
    PhysicsBody( name, nullptr )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElement::ConstElement(const ConstElement& other) :
    PhysicsBody( other )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElement& ConstElement::operator=(const ConstElement& rhs)
{
    PhysicsBody::operator = (rhs);

    return *this;
}
