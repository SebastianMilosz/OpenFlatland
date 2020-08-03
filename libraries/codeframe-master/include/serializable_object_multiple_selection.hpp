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

            smart_ptr<PropertyNode> Property(const std::string& name) override;
            smart_ptr<PropertyNode> PropertyFromPath(const std::string& path) override;

            smart_ptr<ObjectNode> GetNode( unsigned int id = 0U );
            unsigned int GetNodeCount() const override;

            std::string ObjectName( bool idSuffix = true ) const override;
            std::string PathString() const override;

            void Add( smart_ptr<ObjectNode> obj );

            smart_ptr<ObjectSelection> Parent() const override;
            smart_ptr<ObjectSelection> Root() override;
            smart_ptr<ObjectSelection> ObjectFromPath( const std::string& path ) override;
            smart_ptr<ObjectSelection> GetObjectByName( const std::string& name ) override;
            smart_ptr<ObjectSelection> GetObjectById( const uint32_t id ) override;

            /// This method should return true if all objects in selection exist
            bool_t IsValid() const override;
        private:
            void OnDelete(void* deletedPtr);
            std::vector< smart_ptr<ObjectNode> > m_selection;
    };
}

#endif // SERIALIZABLE_OBJECT_MULTIPLE_SELECTION_HPP_INCLUDED
