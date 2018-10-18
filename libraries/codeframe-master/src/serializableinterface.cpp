#include "serializableinterface.hpp"
#include "serializableproperty.hpp"

#include <LoggerUtilities.h>

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableInterface::cSerializableInterface() :
        m_Id( -1 )
    {
        static TypeInitializer typeInitializer;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableInterface::~cSerializableInterface()
    {

    }
}
