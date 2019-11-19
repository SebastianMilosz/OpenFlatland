#ifndef CPATH_HPP_INCLUDED
#define CPATH_HPP_INCLUDED

#include <string>
#include <smartpointer.h>

#include "serializable_object_selection.hpp"
#include "serializable_object_multiple_selection.hpp"

namespace codeframe
{
    class ObjectNode;

    class cPath
    {
        public:
             cPath( ObjectNode& sint );
            ~cPath();

            std::string PathString() const;
            void ParentBound( ObjectNode* parent );
            void ParentUnbound();

            bool IsNameUnique( const std::string& name, const bool checkParent = false ) const;

            smart_ptr<ObjectSelection> Parent() const;
            smart_ptr<ObjectSelection> GetRootObject    ();
            smart_ptr<ObjectSelection> GetObjectFromPath( const std::string& path );
            smart_ptr<ObjectSelection> GetChildByName   ( const std::string& name );

        private:
            ObjectNode& m_sint;
            ObjectNode* m_parent;
    };

}

#endif // CPATH_HPP_INCLUDED
