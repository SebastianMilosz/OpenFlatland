#include "serializableneuronlayercontainer.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuronLayerContainer::SerializableNeuronLayerContainer( std::string name, cSerializableInterface* parent ) :
    cSerializableContainer( name, parent )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuronLayerContainer::~SerializableNeuronLayerContainer()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<codeframe::cSerializableInterface> SerializableNeuronLayerContainer::Create(
                                                     const std::string& className,
                                                     const std::string& objName,
                                                     const std::vector<codeframe::VariantValue>& params
                                                    )
{
    if ( className == "SerializableNeuronLayer" )
    {
    }

    return smart_ptr<codeframe::cSerializableInterface>();
}
