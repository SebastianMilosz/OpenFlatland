#include "serializable_object_node.hpp"
#include "serializable_property.hpp"

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
