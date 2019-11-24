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
            class ObjectSelectionIterator
            {
                friend class ObjectSelection;

                private:
                    ObjectSelectionIterator( ObjectSelection* objectSelection, int id ) :
                        m_id( id ),
                        m_ObjectSelection( objectSelection )
                    {
                    }

                public:
                    bool operator != ( const ObjectSelectionIterator& other ) const
                    {
                        return m_id != other.m_id;
                    }

                    ObjectNode* operator* () const
                    {
                        return m_ObjectSelection->GetNode( m_id );
                    }

                    const ObjectSelectionIterator& operator++()
                    {
                        ++m_id;
                        return *this;
                    }

                private:
                    int m_id;
                    ObjectSelection* m_ObjectSelection;
            };

        public:
                    ObjectSelection( ObjectNode* obj );
           virtual ~ObjectSelection();

            virtual ObjectNode* GetNode( unsigned int id = 0U );
            virtual unsigned int GetNodeCount();

            ObjectSelectionIterator begin()
            {
                return ObjectSelectionIterator( this, 0 );
            }

            ObjectSelectionIterator end()
            {
                return ObjectSelectionIterator( this, GetNodeCount() );
            }

        protected:
            ObjectSelection();

        private:
            ObjectNode* m_selection;
    };
}

#endif // SERIALIZABLE_OBJECT_SELECTION_HPP_INCLUDED
