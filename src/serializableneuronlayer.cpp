#include "serializableneuronlayer.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuronLayer::SerializableNeuronLayer( std::string name, cSerializableInterface* parent ) :
    cSerializableContainer( name, parent )
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
smart_ptr<codeframe::cSerializableInterface> SerializableNeuronLayer::Create(
                                                                   const std::string& className,
                                                                   const std::string& objName,
                                                                   const std::vector<codeframe::VariantValue>& params )
{
    if ( className == "SerializableNeuron" )
    {
    }
}
