#include "neuron_layer_ray.hpp"

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
NeuronLayerRay::NeuronLayerRay( const std::string& name, ObjectNode* parent, const std::string& link ) :
    NeuronLayer( name, parent, link ),
    Data( this, "Data", thrust::host_vector<RayData>(),
           cPropertyInfo().
           Kind(KIND_VECTOR_THRUST_HOST, KIND_RAY_DATA).
           ReferencePath(link).
           Description("Data"))
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void NeuronLayerRay::ProcessData(thrust::host_vector<float>& vectInData, thrust::host_vector<float>& vectOutData)
{

}
