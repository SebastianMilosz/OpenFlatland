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
    Data( this, "Data", thrust::host_vector<float>(),
           cPropertyInfo().
           Kind(KIND_VECTOR_THRUST_HOST, KIND_REAL).
          ReferencePath(link).
           Description("Data"))
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void NeuronLayerVector::ProcessData(thrust::host_vector<float>& vectInData, thrust::host_vector<float>& vectOutData)
{

}
