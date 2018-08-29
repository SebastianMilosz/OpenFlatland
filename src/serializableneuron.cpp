#include "serializableneuron.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuron::SerializableNeuron( std::string name, cSerializableInterface* parent, unsigned int inputCnt ) :
    cSerializable( name, parent ),
    InputsCnt    ( this, "InputsCnt"    , inputCnt               , cPropertyInfo().Kind( KIND_NUMBER ).Description("InputsCnt"), this ),
    InputsWeights( this, "InputsWeights", Vector<float>(inputCnt), cPropertyInfo().Kind( KIND_VECTOR ).Description("InputsWeights"), this ),
    Output       ( this, "Output"       , 0.0F                   , cPropertyInfo().Kind( KIND_REAL   ).Description("Output"), this )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuron::~SerializableNeuron()
{

}
