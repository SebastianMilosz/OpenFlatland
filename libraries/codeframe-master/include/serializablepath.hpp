#ifndef SERIALIZABLEPATH_HPP_INCLUDED
#define SERIALIZABLEPATH_HPP_INCLUDED

#include <string>

namespace codeframe
{
    class ObjectNode;
    class PropertyBase;

    class cSerializablePath
    {
        public:
             cSerializablePath( ObjectNode& sint );
            ~cSerializablePath();

            std::string PathString() const;
            void ParentBound( ObjectNode* parent );
            void ParentUnbound();

            bool IsNameUnique( const std::string& name, const bool checkParent = false ) const;

            ObjectNode*  Parent()     const;
            ObjectNode*  GetRootObject      (                  );
            ObjectNode*  GetObjectFromPath  ( const std::string& path );
            ObjectNode*  GetChildByName     ( const std::string& name );

        private:
            ObjectNode& m_sint;
            ObjectNode* m_parent;
    };

}

#endif // SERIALIZABLEPATH_HPP_INCLUDED
