#include "referencemanager.hpp"

#include <serializable.hpp>
#include <utilities/LoggerUtilities.h>

namespace codeframe
{

std::map<std::string, PropertyBase*> ReferenceManager::m_referencePathMap;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ReferenceManager::ReferenceManager() :
    m_referencePath(""),
    m_property( NULL )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ReferenceManager::~ReferenceManager()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ReferenceManager::SetReference( const std::string& refPath, PropertyBase* prop )
{
    m_referencePath = refPath;
    m_property = prop;

    if ( m_referencePath.size() != 0 )
    {
        if ( NULL != m_property )
        {
            std::string referenceAbsolutePath = PreparePath( m_referencePath, m_property );
            m_referencePathMap.insert( std::pair<std::string, PropertyBase*>(referenceAbsolutePath,m_property) );
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ReferenceManager::SetProperty( PropertyBase* prop )
{
    if ( (m_referencePath.size() != 0) && (NULL != prop) && (NULL == m_property) )
    {
        m_property = prop;
        std::string referenceAbsolutePath = PreparePath( m_referencePath, m_property );
        m_referencePathMap.insert( std::pair<std::string, PropertyBase*>(referenceAbsolutePath,m_property) );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const std::string& ReferenceManager::Get() const
{
    return m_referencePath;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ReferenceManager::LogUnresolvedReferences()
{
    std::map<std::string, PropertyBase*>::iterator it;

    for ( it = m_referencePathMap.begin(); it != m_referencePathMap.end(); it++ )
    {
        PropertyBase* prop = it->second;
        if ( (PropertyBase*)NULL != prop )
        {
            cSerializableInterface* propertyParent = prop->Parent();
            std::string propertyParentPath = "NULL";

            if ( NULL != propertyParent )
            {
                propertyParentPath = propertyParent->Path().PathString();
            }

            LOGGER( LOG_INFO << "Unresolved reference to: " << it->first << " from object: " << propertyParentPath << "." << prop->Name() );
        }
        else
        {
            LOGGER( LOG_ERROR << "Unresolved reference to: " << it->first << " from object: NULL" );
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string ReferenceManager::PreparePath( const std::string& path, PropertyBase* prop )
{
    cSerializableInterface* propertyParent = prop->Parent();

    std::string retString = std::string( path );

    if ( NULL != propertyParent )
    {
        bool isDownHierarchy = (strncmp(retString.c_str(), "..", strlen("..")) == 0);
        bool isRelative = (strncmp(retString.c_str(), "/", strlen("/")) == 0);

        // We have to make path absolute
        if ( isDownHierarchy )
        {
            retString.erase(0, retString.find("/"));
        }

        if ( isRelative || isDownHierarchy )
        {
            std::string propertyPath = propertyParent->Path().PathString();
            retString = propertyPath + retString;
        }
    }

    return retString;
}

}
