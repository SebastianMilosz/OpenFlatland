#include "serializableneuronlayer.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuronLayer::SerializableNeuronLayer( const std::string& name, cSerializableInterface* parent ) :
    cSerializable( name, parent ),
    Activation      ( this, "Activation"      , 0                            , cPropertyInfo().Kind( KIND_ENUM   ).Enum("Identity,Binary step,Logistic").Description("Activation Function")),
    WeightDimensions( this, "WeightDimensions", std::vector<unsigned int>(0) , cPropertyInfo().Kind( KIND_VECTOR ).Description("WeightDimensions") ),
    WeightMatrix    ( this, "WeightMatrix"    , thrust::host_vector<float>(0), cPropertyInfo().Kind( KIND_VECTOR ).Description("WeightMatrix") ),
    Input           ( this, "Input"           , thrust::host_vector<float>(0), cPropertyInfo().Kind( KIND_VECTOR ).Description("Input") ),
    Output          ( this, "Output"          , thrust::host_vector<float>(0), cPropertyInfo().Kind( KIND_VECTOR ).Description("Output") ),
    WeightDimensionsCnt( 0U ),
    WeightMatrixCnt( 0U ),
    InputCnt( 0U ),
    OutputCnt( 0U )
{

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
    // Preprocessing if needed, prepare data matrix
    if ( true == NeedRecreateInternalState() )
    {
        RecreateInternalState();
    }


}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool SerializableNeuronLayer::NeedRecreateInternalState()
{
    if ( InputCnt  != Input.GetBaseValue().size() ) { return true; }
    if ( OutputCnt != Output.GetBaseValue().size() ) { return true; }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SerializableNeuronLayer::RecreateInternalState()
{

}
