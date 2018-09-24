#ifndef REFERENCEMANAGER_HPP_INCLUDED
#define REFERENCEMANAGER_HPP_INCLUDED

#include <map>

namespace codeframe
{
    class cSerializableInterface;

    class ReferenceManager
    {
        public:
            ReferenceManager();
            ~ReferenceManager();

            void Set( const std::string& refPath );
            const std::string& Get() const;
        private:
            std::string m_referencePath;
            static std::map<std::string, cSerializableInterface*> m_referencePathMap;
    };
}

#endif // REFERENCEMANAGER_HPP_INCLUDED
