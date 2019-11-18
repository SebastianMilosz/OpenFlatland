#ifndef SERIALIZABLE_OBJECT_SELECTION_HPP_INCLUDED
#define SERIALIZABLE_OBJECT_SELECTION_HPP_INCLUDED

#include <string>
#include <vector>

namespace codeframe
{
    class ObjectNode;

     /*****************************************************************************
     * @class This class stores Object's selection
     *****************************************************************************/
    class ObjectSelection
    {
        public:
            ObjectSelection( ObjectNode* obj );
           ~ObjectSelection();

            ObjectNode* GetNode( unsigned int id = 0U );

        private:
            ObjectNode* m_selection;
    };
}

#endif // SERIALIZABLE_OBJECT_SELECTION_HPP_INCLUDED
