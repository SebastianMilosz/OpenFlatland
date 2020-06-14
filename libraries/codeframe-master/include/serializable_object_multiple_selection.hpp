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
            using ObjectSelectionIterator = ObjectSelection::ObjectSelectionIterator;

            ObjectMultipleSelection();
            ObjectMultipleSelection( smart_ptr<ObjectNode> obj );
           ~ObjectMultipleSelection();

            smart_ptr<ObjectNode> GetNode( unsigned int id = 0U );
            unsigned int GetNodeCount();

            std::string ObjectName( bool idSuffix = true ) const;

            void Add( smart_ptr<ObjectNode> obj );

        private:
            std::vector< smart_ptr<ObjectNode> > m_selection;
    };
}

#endif // SERIALIZABLE_OBJECT_MULTIPLE_SELECTION_HPP_INCLUDED
