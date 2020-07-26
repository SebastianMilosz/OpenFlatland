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
    Input( this, "Input", thrust::host_vector<RayData>(),
           cPropertyInfo().
           Kind(KIND_VECTOR_THRUST_HOST, KIND_RAY_DATA).
           ReferencePath("../../Vision.VisionVector").
           Description("Input"))
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void NeuronLayerRay::Calculate()
{

}
