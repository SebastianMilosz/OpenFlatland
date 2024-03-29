#include "entity.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Entity::Entity( const std::string& name, int x, int y ) : EntityGhost( name, x, y )
{
    //ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Entity::Entity(const Entity& other) : EntityGhost( other )
{
    //copy ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Entity& Entity::operator=(const Entity& rhs)
{
    if (this == &rhs) return *this; // handle self assignment
    //assignment operator
    return *this;
}
