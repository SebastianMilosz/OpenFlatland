#include "serializableneuron.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuron::SerializableNeuron( std::string name, cSerializableInterface* parent, unsigned int inputCnt ) :
    cSerializable( name, parent ),
    InputsWeights( this, "Weights", std::vector<float>(), cPropertyInfo().Kind( KIND_VECTOR ).Description("Weights"), this ),
    Output       ( this, "Output" , 0.0F                , cPropertyInfo().Kind( KIND_REAL   ).Description("Output"), this )
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
