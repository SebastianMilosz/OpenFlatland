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
    cSerializableInterface::cSerializableInterface()
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
