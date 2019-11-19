#ifndef SERIALIZABLE_OBJECT_MULTIPLE_SELECTION_HPP_INCLUDED
#define SERIALIZABLE_OBJECT_MULTIPLE_SELECTION_HPP_INCLUDED

#include "serializable_object_selection.hpp"

#include <string>
#include <vector>

namespace codeframe
{
    class ObjectNode;

     /*****************************************************************************
     * @class This class stores Object's selections
     *****************************************************************************/
    class ObjectMultipleSelection : public ObjectSelection
    {
        public:
            ObjectMultipleSelection( ObjectNode* obj );
           ~ObjectMultipleSelection();

            ObjectNode* GetNode( unsigned int id = 0U );
            unsigned int GetNodeCount();

            void Add( ObjectNode* obj );

        private:
            std::vector<ObjectNode*> m_selection;
    };
}

#endif // SERIALIZABLE_OBJECT_MULTIPLE_SELECTION_HPP_INCLUDED
