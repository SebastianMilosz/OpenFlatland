#ifndef REFERENCEMANAGER_HPP_INCLUDED
#define REFERENCEMANAGER_HPP_INCLUDED

#include <list>
#include <string>

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
            static std::list<std::string> m_referencePathList;
    };
}

#endif // REFERENCEMANAGER_HPP_INCLUDED
