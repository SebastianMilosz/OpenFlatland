#include "entityghost.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityGhost::EntityGhost( std::string name, int x, int y, int z ) : EntityShell( name, x, y, z )
{
    //ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityGhost::~EntityGhost()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityGhost::EntityGhost(const EntityGhost& other) :
    EntityShell( other )
{
    //copy ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityGhost& EntityGhost::operator=(const EntityGhost& rhs)
{
    if (this == &rhs) return *this; // handle self assignment
    //assignment operator
    return *this;
}
