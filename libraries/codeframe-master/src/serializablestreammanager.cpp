#include <serializablestreammanager.h>

namespace codeframe
{

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSerializableStreamManager::cSerializableStreamManager( cSerializable* parentObject ) : cSerializableContainer( "StreamManager", parentObject ),
    Enable( this, "ENABLE" , false, cPropertyInfo().Kind(KIND_LOGIC).Description("Enable Processing") )
{
    Enable.signalChanged.connect(this, &cSerializableStreamManager::OnEnable);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSerializableStreamManager::~cSerializableStreamManager()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableStreamManager::OnEnable( Property* prop )
{

}

/*****************************************************************************/
/**
  * @brief
  * 
 **
******************************************************************************/
cSerializableStream* cSerializableStreamManager::Create( std::string className, std::string objName, int cnt )
{
	if( className == "cSerializableStream" )
    {
		cSerializableStream* retObj = new cSerializableStream( objName, (cSerializable*)this ); 
        InsertObject( retObj );
        return retObj;
    }
	return NULL;
}

}
