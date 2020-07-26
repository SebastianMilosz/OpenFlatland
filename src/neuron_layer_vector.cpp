#include "neuron_layer_vector.hpp"

#include <utilities/MathUtilities.h>
#include <utilities/LoggerUtilities.h>

#include <iostream>
#include <random>
#include <chrono>

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
NeuronLayerVector::NeuronLayerVector( const std::string& name, ObjectNode* parent, const std::string& link ) :
    NeuronLayer( name, parent, link ),
    Input( this, "Input", thrust::host_vector<float>(),
           cPropertyInfo().
           Kind(KIND_VECTOR_THRUST_HOST, KIND_REAL).
          ReferencePath("../../Energy.EnergyVector").
           Description("Input"))
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void NeuronLayerVector::Calculate()
{

}
