#ifndef SERIALIZABLE_OBJECT_SELECTION_HPP_INCLUDED
#define SERIALIZABLE_OBJECT_SELECTION_HPP_INCLUDED

#include <string>
#include <vector>
#include <smartpointer.h>

namespace codeframe
{
    class ObjectNode;
    class PropertyNode;

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

                    smart_ptr<ObjectNode> operator* () const
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
                    ObjectSelection( smart_ptr<ObjectNode> obj );
           virtual ~ObjectSelection();

            virtual smart_ptr<PropertyNode> Property(const std::string& name);
            virtual smart_ptr<PropertyNode> PropertyFromPath(const std::string& path);

            virtual unsigned int GetNodeCount();

            virtual std::string ObjectName( bool idSuffix = true ) const;
            virtual std::string PathString() const;

            virtual smart_ptr<ObjectSelection> Parent() const;
            virtual smart_ptr<ObjectSelection> Root();
            virtual smart_ptr<ObjectSelection> ObjectFromPath( const std::string& path );
            virtual smart_ptr<ObjectSelection> GetObjectByName( const std::string& name );
            virtual smart_ptr<ObjectSelection> GetObjectById( const uint32_t id );

            ObjectSelectionIterator begin()
            {
                return ObjectSelectionIterator( this, 0 );
            }

            ObjectSelectionIterator end()
            {
                return ObjectSelectionIterator( this, GetNodeCount() );
            }

            virtual smart_ptr<ObjectNode> GetNode( unsigned int id = 0U );
        protected:
            ObjectSelection();

        private:
            smart_ptr<ObjectNode> m_selection;
    };
}

#endif // SERIALIZABLE_OBJECT_SELECTION_HPP_INCLUDED
