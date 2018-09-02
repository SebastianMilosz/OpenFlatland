#include "serializableneuronlayercontainer.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuronLayerContainer::SerializableNeuronLayerContainer( std::string name, cSerializableInterface* parent ) :
    cSerializableContainer( name, parent ),
    LayersCnt( this, "LayersCnt" , 1U , cPropertyInfo().Kind( KIND_NUMBER ).Description("LayersCnt"), this, NULL, &SerializableNeuronLayerContainer::SetLayersCnt )
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
        smart_ptr<SerializableNeuronLayer> obj = smart_ptr<SerializableNeuronLayer>( new SerializableNeuronLayer( objName, NULL ) );

        (void)InsertObject( obj );

        return obj;
    }

    return smart_ptr<codeframe::cSerializableInterface>();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SerializableNeuronLayerContainer::SetLayersCnt( unsigned int cnt )
{
    CreateRange( "SerializableNeuronLayer", "AnnLayer", cnt );
}
