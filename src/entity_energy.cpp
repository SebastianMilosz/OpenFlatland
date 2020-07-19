#include "entity_energy.hpp"

#include <cmath>

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityEnergy::EntityEnergy(codeframe::ObjectNode* parent) :
    PhysicsBody("Energy", parent),
    Energy( this, "Energy"  , 0.0F , cPropertyInfo().Kind(KIND_REAL).Description("Energy") )
{
    //ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityEnergy::synchronize(b2Body& body)
{

}
