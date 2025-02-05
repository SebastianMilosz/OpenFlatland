#ifndef PROPERTYIGNORELIST_HPP_INCLUDED
#define PROPERTYIGNORELIST_HPP_INCLUDED

#include "serializable_object_node.hpp"

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
            sIgnoreEntry() : Name(""), ClassName(""), BuildType(STATIC), Ignore(false) {}
            sIgnoreEntry( const std::string& name, const std::string& className = "", eBuildType buildType = STATIC, bool ignore = true) :
                Name(name), ClassName(className), BuildType(buildType), Ignore(ignore) {}

            std::string Name;
            std::string ClassName;
            eBuildType  BuildType;
            bool Ignore;
        };

        void AddToList( ObjectNode* serObj, bool ignore = true )
        {
            if( serObj )
            {
                m_vectorIgnoreEntry.push_back( sIgnoreEntry( serObj->Identity().ObjectName(), serObj->Class(), serObj->BuildType(), ignore ) );
            }
        }

        void AddToList( const std::string& name = "", const std::string& className = "", eBuildType buildType = STATIC, bool ignore = true )
        {
            m_vectorIgnoreEntry.push_back( sIgnoreEntry( name, className, buildType, ignore ) );
        }

        bool IsIgnored( ObjectNode* serObj ) const
        {
            if( m_vectorIgnoreEntry.empty() == false && serObj )
            {
                for( std::vector<sIgnoreEntry>::const_iterator it = m_vectorIgnoreEntry.cbegin(); it != m_vectorIgnoreEntry.cend(); ++it )
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
