#include "serializableneuronlayer.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuronLayer::SerializableNeuronLayer( std::string name, cSerializableInterface* parent ) :
    cSerializable( name, parent ),
    NeuronCnt( this, "NeuronCnt" , 10U                  , cPropertyInfo().Kind( KIND_NUMBER ).Description("NeuronCnt"), this, NULL, &SerializableNeuronLayer::SetNeuronCnt),
    Input    ( this, "Input"     , std::vector<float>(0), cPropertyInfo().Kind( KIND_VECTOR ).Description("Input") ),
    Output   ( this, "Output"    , std::vector<float>(0), cPropertyInfo().Kind( KIND_VECTOR ).Description("Output") )
{
    SetNeuronCnt( (unsigned int)NeuronCnt );
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
void SerializableNeuronLayer::SetNeuronCnt( unsigned int cnt )
{

}
