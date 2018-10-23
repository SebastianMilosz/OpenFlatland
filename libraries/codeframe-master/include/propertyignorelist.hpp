#ifndef PROPERTYIGNORELIST_HPP_INCLUDED
#define PROPERTYIGNORELIST_HPP_INCLUDED

#include "serializableinterface.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    class cIgnoreList
    {
    public:
        cIgnoreList() { }

        struct sIgnoreEntry
        {
            sIgnoreEntry() : Name(""), ClassName(""), BuildType(BUILD_TYPE_STATIC), Ignore(false) {}
            sIgnoreEntry(std::string name, std::string className = "", eBuildType buildType = BUILD_TYPE_STATIC, bool ignore = true) :
                Name(name), ClassName(className), BuildType(buildType), Ignore(ignore) {}

            std::string Name;
            std::string ClassName;
            eBuildType  BuildType;
            bool Ignore;
        };

        void AddToList( cSerializableInterface* serObj, bool ignore = true )
        {
            if( serObj )
            {
                m_vectorIgnoreEntry.push_back( sIgnoreEntry( serObj->Identity().ObjectName(), serObj->Class(), serObj->BuildType(), ignore ) );
            }
        }

        void AddToList( std::string name = "", std::string className = "", eBuildType buildType = BUILD_TYPE_STATIC, bool ignore = true )
        {
            m_vectorIgnoreEntry.push_back( sIgnoreEntry( name, className, buildType, ignore ) );
        }

        bool IsIgnored( cSerializableInterface* serObj )
        {
            if( m_vectorIgnoreEntry.empty() == false && serObj )
            {
                for( std::vector<sIgnoreEntry>::iterator it = m_vectorIgnoreEntry.begin(); it != m_vectorIgnoreEntry.end(); ++it )
                {
                    sIgnoreEntry entry = *it;

                    /* std::cout << *it; ... */
                    if( entry.Name == serObj->Identity().ObjectName() && entry.BuildType == serObj->BuildType() && entry.Ignore )
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        void Dispose()
        {
            m_vectorIgnoreEntry.clear();
        }

    private:
        std::vector<sIgnoreEntry> m_vectorIgnoreEntry;

    };
}

#endif // PROPERTYIGNORELIST_HPP_INCLUDED
