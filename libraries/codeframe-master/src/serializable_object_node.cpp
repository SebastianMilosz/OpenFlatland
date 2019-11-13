#include "serializable_object_node.hpp"
#include "serializableproperty.hpp"

#include <LoggerUtilities.h>

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    ObjectNode::ObjectNode()
    {
        static TypeInitializer typeInitializer;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    ObjectNode::~ObjectNode()
    {

    }
}
