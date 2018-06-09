#include "serializableinterface.h"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
float cSerializableInterface::LibraryVersion()
{
    return 0.2;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string cSerializableInterface::LibraryVersionString()
{
    return std::string( "Serializable library version 0.2" );
}
