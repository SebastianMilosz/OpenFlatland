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
        m_dummyProperty(NULL, "DUMMY", TYPE_NON, cPropertyInfo()),
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
