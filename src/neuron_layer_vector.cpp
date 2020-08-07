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
           Description("Data")),
    Max(this, "Max", 0.0f, cPropertyInfo().Kind( KIND_REAL ).Description("Max"), [this]() -> const float& { return this->m_Max; }),
    Min(this, "Min", 0.0f, cPropertyInfo().Kind( KIND_REAL ).Description("Min"), [this]() -> const float& { return this->m_Min; }),
    Evg(this, "Evg", 0.0f, cPropertyInfo().Kind( KIND_REAL ).Description("Evg"), [this]() -> const float& { return this->m_Evg; }),
    m_Max(std::numeric_limits<float>::min()),
    m_Min(std::numeric_limits<float>::max()),
    m_Evg(0.0f)
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void NeuronLayerVector::ProcessData(thrust::host_vector<float>& vectData)
{
    float tmpMax = std::numeric_limits<float>::min();
    float tmpMin = std::numeric_limits<float>::max();

    thrust::host_vector<float>& internalVector = Data.GetValue();
    thrust::for_each(internalVector.begin(), internalVector.end(), copy_functor(vectData, tmpMax, tmpMin));
    thrust::for_each(vectData.begin()      , vectData.end()      , normalize_functor(tmpMax, tmpMin));

    m_Max = tmpMax;
    m_Min = tmpMin;
}
