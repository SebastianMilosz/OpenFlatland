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
    SetLayersCnt( (unsigned int)LayersCnt );
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
    unsigned int thisCnt = Count();
    // Set layer cnt to be at least configured
    if ( cnt > thisCnt )
    {
        unsigned int newCnt = (cnt - thisCnt);
        CreateRange( "SerializableNeuronLayer", "AnnLayer", newCnt );
    }
}
