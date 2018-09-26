#ifndef REFERENCEMANAGER_HPP_INCLUDED
#define REFERENCEMANAGER_HPP_INCLUDED

#include <map>
#include <string>

namespace codeframe
{
    class cSerializableInterface;

    class ReferenceManager
    {
        public:
            ReferenceManager();
            ~ReferenceManager();

            void SetReference( const std::string& refPath, cSerializableInterface* obj = NULL );
            void SetParent( cSerializableInterface* obj );
            const std::string& Get() const;

            static void LogUnresolvedReferences();
        private:
            std::string m_referencePath;
            cSerializableInterface* m_parent;
            static std::map<std::string, cSerializableInterface*> m_referencePathMap;
    };
}

#endif // REFERENCEMANAGER_HPP_INCLUDED
