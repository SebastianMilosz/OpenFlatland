#ifndef SERIALIZABLEPATH_HPP_INCLUDED
#define SERIALIZABLEPATH_HPP_INCLUDED

#include <string>

namespace codeframe
{
    class cSerializableInterface;
    class PropertyBase;

    class cSerializablePath
    {
        public:
             cSerializablePath( cSerializableInterface& sint );
            ~cSerializablePath();

            std::string PathString() const;
            void ParentBound( cSerializableInterface* parent );
            void ParentUnbound();

            bool IsNameUnique( const std::string& name, bool checkParent = false ) const;

            cSerializableInterface*  Parent()     const;
            cSerializableInterface*  GetRootObject      (                  );
            cSerializableInterface*  GetObjectFromPath  ( const std::string& path );
            cSerializableInterface*  GetChildByName     ( const std::string& name );

        private:
            cSerializableInterface& m_sint;
            cSerializableInterface* m_parent;
    };

}

#endif // SERIALIZABLEPATH_HPP_INCLUDED
