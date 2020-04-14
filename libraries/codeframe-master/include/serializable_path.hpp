#ifndef CPATH_HPP_INCLUDED
#define CPATH_HPP_INCLUDED

#include <string>
#include <smartpointer.h>
#include <typedefs.hpp>

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
            bool_t ParentBound( ObjectNode* parent );
            void ParentUnbound();

            bool_t IsNameUnique( const std::string& name, const bool_t checkParent = false ) const;

            smart_ptr<ObjectSelection> Parent() const;
            smart_ptr<ObjectSelection> GetRootObject    ();
            smart_ptr<ObjectSelection> GetObjectFromPath( const std::string& path );
            smart_ptr<ObjectSelection> GetChildByName   ( const std::string& name );

            static std::string PreparePath(const std::string& path, std::vector<std::string>& pathDir, smart_ptr<ObjectSelection> propertyParent);

        private:
            static bool_t IsDownHierarchy(const std::string& path);
            static bool_t IsRelativeHierarchy(const std::string& path);

            ObjectNode& m_sint;
            ObjectNode* m_parent;
    };

}

#endif // CPATH_HPP_INCLUDED
