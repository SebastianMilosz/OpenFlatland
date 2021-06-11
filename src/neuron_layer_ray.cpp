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
void NeuronLayerRay::GiveData(thrust::host_vector<float>& vectData)
{
    float tmpMaxDistance = std::numeric_limits<float>::min();
    float tmpMinDistance = std::numeric_limits<float>::max();
    float tmpMaxFixture  = std::numeric_limits<float>::min();
    float tmpMinFixture  = std::numeric_limits<float>::max();

    thrust::host_vector<RayData>& internalVector = Data.GetValue();

    thrust::host_vector<float> vectDistanceData;
    thrust::host_vector<float> vectFixtureData;

    vectDistanceData.reserve(internalVector.size());
    vectFixtureData.reserve(internalVector.size());

    thrust::for_each(
                         internalVector.begin(),
                         internalVector.end(),
                         copy_functor<Layer::LAYER_DISTANCE>(vectDistanceData, tmpMaxDistance, tmpMinDistance)
                    );
    thrust::for_each(
                         vectDistanceData.begin(),
                         vectDistanceData.end(),
                         normalize_functor(tmpMaxDistance, tmpMinDistance)
                    );

    thrust::for_each(
                         internalVector.begin(),
                         internalVector.end(),
                         copy_functor<Layer::LAYER_FIXTURE>(vectFixtureData, tmpMaxFixture, tmpMinFixture)
                    );
    thrust::for_each(
                         vectFixtureData.begin(),
                         vectFixtureData.end(),
                         normalize_functor(tmpMaxFixture, tmpMinFixture)
                    );

    thrust::counting_iterator<uint32_t> first(0U);
    thrust::counting_iterator<uint32_t> last = first + internalVector.size();

    thrust::for_each(
                         thrust::make_zip_iterator(thrust::make_tuple(
                                                                      first,
                                                                      vectDistanceData.begin(),
                                                                      vectFixtureData.begin()
                                                                     )),
                         thrust::make_zip_iterator(thrust::make_tuple(
                                                                      last,
                                                                      vectDistanceData.end(),
                                                                      vectFixtureData.end()
                                                                     )),
                        marge_functor(vectData)
                    );

    m_MaxDistance = tmpMaxDistance;
    m_MinDistance = tmpMinDistance;
    m_MaxFixture  = tmpMaxFixture;
    m_MinFixture  = tmpMinFixture;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void NeuronLayerRay::TakeData(thrust::host_vector<float>& vectData)
{

}
