#include "constelement.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElement::ConstElement( std::string name ) :
    cSerializable( name, NULL )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElement::~ConstElement()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElement::ConstElement(const ConstElement& other) :
    cSerializable( other )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElement& ConstElement::operator=(const ConstElement& other)
{

}
