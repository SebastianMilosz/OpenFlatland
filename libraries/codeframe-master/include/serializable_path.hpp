#ifndef CPATH_HPP_INCLUDED
#define CPATH_HPP_INCLUDED

#include <string>

namespace codeframe
{
    class ObjectNode;
    class PropertyBase;

    class cPath
    {
        public:
             cPath( ObjectNode& sint );
            ~cPath();

            std::string PathString() const;
            void ParentBound( ObjectNode* parent );
            void ParentUnbound();

            bool IsNameUnique( const std::string& name, const bool checkParent = false ) const;

            ObjectNode*  Parent()     const;
            ObjectNode*  GetRootObject    (                  );
            ObjectNode*  GetObjectFromPath( const std::string& path );
            ObjectNode*  GetChildByName   ( const std::string& name );

        private:
            ObjectNode& m_sint;
            ObjectNode* m_parent;
    };

}

#endif // CPATH_HPP_INCLUDED
