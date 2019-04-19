#include "serializableneuronlayer.hpp"

#include <utilities/MathUtilities.h>
#include <utilities/LoggerUtilities.h>

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuronLayer::SerializableNeuronLayer( const std::string& name, cSerializableInterface* parent ) :
    cSerializable( name, parent ),
    Activation      ( this, "Activation"      , 0                            , cPropertyInfo().Kind( KIND_ENUM   ).Enum("Identity,Binary step,Logistic").Description("Activation Function")),
    WeightDimensions( this, "WeightDimensions", std::vector<unsigned int>(0) , cPropertyInfo().Kind( KIND_VECTOR ).Description("WeightDimensions"), this, &SerializableNeuronLayer::GetWeightDimensionsVector ),
    WeightMatrix    ( this, "WeightMatrix"    , thrust::host_vector<float>(0), cPropertyInfo().Kind( KIND_VECTOR ).Description("WeightMatrix") ),
    Input           ( this, "Input"           , thrust::host_vector<float>(0), cPropertyInfo().Kind( KIND_VECTOR ).Description("Input") ),
    Output          ( this, "Output"          , thrust::host_vector<float>(0), cPropertyInfo().Kind( KIND_VECTOR ).Description("Output") )
{
    // Signal On property change connection
    WeightDimensions.signalChanged.connect( this, &SerializableNeuronLayer::OnWeightDimensionsVectorChanged );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuronLayer::~SerializableNeuronLayer()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SerializableNeuronLayer::Calculate()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool SerializableNeuronLayer::InitializeNetwork()
{
    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SerializableNeuronLayer::OnWeightDimensionsVectorChanged( codeframe::PropertyBase* prop )
{
    if ( InitializeNetwork() == true )
    {
        LOGGER_DEBUG( LOG_INFO << LOG_LEVEL7 << prop->Path() << " change has triggered Serializable Neuron Layer Initialization" );
    }
    else
    {
        LOGGER( LOG_ERROR << prop->Path() << " Serializable Neuron Layer Initialization Fail!" );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const std::vector<unsigned int>& SerializableNeuronLayer::GetWeightDimensionsVector()
{
    return m_WeightDimensions;
}
