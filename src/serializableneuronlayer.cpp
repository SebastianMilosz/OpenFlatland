#include "serializableneuronlayer.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuronLayer::SerializableNeuronLayer( std::string name, cSerializableInterface* parent ) :
    cSerializable( name, parent ),
    Activation      ( this, "Activation"      , 0                           , cPropertyInfo().Kind( KIND_ENUM   ).Enum("Identity,Binary step,Logistic").Description("Activation Function")),
    WeightDimensions( this, "WeightDimensions", std::vector<unsigned int>(0), cPropertyInfo().Kind( KIND_VECTOR ).Description("WeightDimensions"), this, NULL, &SerializableNeuronLayer::SetNeuronCnt),
    WeightMatrix    ( this, "WeightMatrix"    , std::vector<float>(0)       , cPropertyInfo().Kind( KIND_VECTOR ).Description("WeightMatrix") ),
    Input           ( this, "Input"           , std::vector<float>(0)       , cPropertyInfo().Kind( KIND_VECTOR ).Description("Input") ),
    Output          ( this, "Output"          , std::vector<float>(0)       , cPropertyInfo().Kind( KIND_VECTOR ).Description("Output") )
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

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SerializableNeuronLayer::SetNeuronCnt( std::vector<unsigned int> cntVec )
{

}
