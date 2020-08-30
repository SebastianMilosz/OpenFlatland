#include "neuron_layer_ray.hpp"

#include <utilities/MathUtilities.h>
#include <utilities/LoggerUtilities.h>

#include <iostream>
#include <random>
#include <chrono>
#include <limits>

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
           Description("Data")),
    MaxDistance(this, "MaxDistance", 0.0f, cPropertyInfo().Kind( KIND_REAL ).Description("MaxDistance"), [this]() -> const float& { return this->m_MaxDistance; }),
    MinDistance(this, "MinDistance", 0.0f, cPropertyInfo().Kind( KIND_REAL ).Description("MinDistance"), [this]() -> const float& { return this->m_MinDistance; }),
    MaxFixture (this, "MaxFixture" , 0.0f, cPropertyInfo().Kind( KIND_REAL ).Description("MaxFixture") , [this]() -> const float& { return this->m_MaxFixture; }),
    MinFixture (this, "MinFixture" , 0.0f, cPropertyInfo().Kind( KIND_REAL ).Description("MinFixture") , [this]() -> const float& { return this->m_MinFixture; }),
    m_MaxDistance(std::numeric_limits<float>::min()),
    m_MinDistance(std::numeric_limits<float>::max()),
    m_MaxFixture(std::numeric_limits<float>::min()),
    m_MinFixture(std::numeric_limits<float>::max())
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void NeuronLayerRay::ProcessData(thrust::host_vector<float>& vectData)
{
    float tmpMaxDistance = std::numeric_limits<float>::min();
    float tmpMinDistance = std::numeric_limits<float>::max();
    float tmpMaxFixture  = std::numeric_limits<float>::min();
    float tmpMinFixture  = std::numeric_limits<float>::max();

    auto normalizeBegin = std::distance(vectData.begin(), vectData.end());

    thrust::host_vector<RayData>& internalVector = Data.GetValue();
    thrust::for_each(internalVector.begin(), internalVector.end(), copy_functor<Layer::LAYER_DISTANCE>(vectData, tmpMaxDistance, tmpMinDistance));
    thrust::for_each(std::next(vectData.begin(), normalizeBegin), vectData.end(), normalize_functor(tmpMaxDistance, tmpMinDistance));

    normalizeBegin = std::distance(vectData.begin(), vectData.end());
    thrust::for_each(internalVector.begin(), internalVector.end(), copy_functor<Layer::LAYER_FIXTURE>(vectData, tmpMaxFixture, tmpMinFixture));
    thrust::for_each(std::next(vectData.begin(), normalizeBegin), vectData.end(), normalize_functor(tmpMaxFixture, tmpMinFixture));

    m_MaxDistance = tmpMaxDistance;
    m_MinDistance = tmpMinDistance;
    m_MaxFixture  = tmpMaxFixture;
    m_MinFixture  = tmpMinFixture;
}
