#ifndef REFERENCEMANAGER_HPP_INCLUDED
#define REFERENCEMANAGER_HPP_INCLUDED

#include <map>
#include <string>

namespace codeframe
{
    class PropertyBase;
    class cSerializableInterface;

    class ReferenceManager
    {
        public:
            ReferenceManager();
            ~ReferenceManager();

            void SetReference( const std::string& refPath, PropertyBase* prop = NULL );
            void SetProperty( PropertyBase* prop );
            const std::string& Get() const;

            static void ResolveReferences( cSerializableInterface& root );

            static void LogUnresolvedReferences();
        private:
            static std::string PreparePath( const std::string& path, PropertyBase* prop );

            std::string m_referencePath;
            PropertyBase* m_property;
            static std::map<std::string, PropertyBase*> m_referencePathMap;
    };
}

#endif // REFERENCEMANAGER_HPP_INCLUDED
