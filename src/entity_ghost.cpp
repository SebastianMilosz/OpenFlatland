#include "entity_ghost.hpp"

#include <LoggerUtilities.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityGhost::EntityGhost( const std::string& name, int x, int y ) :
    EntityShell( name, x, y ),
    m_NeuronEngine( "NeuronEngine", this )
{
    //ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityGhost::EntityGhost(const EntityGhost& other) :
    EntityShell( other ),
    m_NeuronEngine( "NeuronEngine", this )
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

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityGhost::CalculateNeuralNetworks()
{
    m_NeuronEngine.Calculate();
}
