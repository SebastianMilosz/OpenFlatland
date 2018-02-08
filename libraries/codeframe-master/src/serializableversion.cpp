#include "serializableinterface.h"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
float cSerializableInterface::Version()
{
    return 0.2;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string cSerializableInterface::VersionString()
{
    return std::string( "Serializable library version 0.2" );
}
