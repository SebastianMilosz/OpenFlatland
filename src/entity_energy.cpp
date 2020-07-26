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
    Energy      ( this, "Energy"      , 0.0F                        , cPropertyInfo().Kind(KIND_REAL).Description("Energy") ),
    EnergyVector( this, "EnergyVector", thrust::host_vector<float>(), cPropertyInfo().Kind(KIND_VECTOR_THRUST_HOST, KIND_REAL).Description("EnergyVector"), std::bind(&EntityEnergy::GetConstEnergyVector, this), nullptr, std::bind(&EntityEnergy::GetEnergyVector, this) ),
    m_energyDataVector(1U, 0.0f)
{
    //ctor
}


/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const thrust::host_vector<float>& EntityEnergy::GetConstEnergyVector() const
{
    return m_energyDataVector;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
thrust::host_vector<float>& EntityEnergy::GetEnergyVector()
{
    return m_energyDataVector;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityEnergy::synchronize(b2Body& body)
{
    m_energyDataVector[0U] = (float)Energy;
}
