#include "referencemanager.hpp"

#include <cstring>      // std::strlen

#include <serializable.hpp>
#include <utilities/LoggerUtilities.h>

namespace codeframe
{

std::map<std::string, ReferenceManager::sReferenceData> ReferenceManager::m_referencePathMap;

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

    if ( m_referencePath != "" )
    {
        if ( NULL != m_property )
        {
            std::string referenceAbsolutePath = PreparePath( m_referencePath, m_property );
            sReferenceData refData;
            refData.Property = m_property;
            refData.RefPath = m_referencePath;
            m_referencePathMap.insert( std::pair<std::string, sReferenceData>(referenceAbsolutePath, refData) );
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
    if ( (m_referencePath != "") && (NULL != prop) && (NULL == m_property) )
    {
        m_property = prop;
        std::string referenceAbsolutePath = PreparePath( m_referencePath, m_property );
        sReferenceData refData;
        refData.Property = m_property;
        refData.RefPath = m_referencePath;
        m_referencePathMap.insert( std::pair<std::string, sReferenceData>(referenceAbsolutePath, refData) );
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
void ReferenceManager::ResolveReferences( ObjectNode& root )
{
    std::map<std::string, sReferenceData>::iterator it;

    for ( it = m_referencePathMap.begin(); it != m_referencePathMap.end();  )
    {
        sReferenceData refData = it->second;
        std::string path = it->first;

        std::string referenceAbsolutePath = PreparePath( refData.RefPath, refData.Property );

        if ( (PropertyBase*)NULL != refData.Property )
        {
            smart_ptr<PropertyNode> targetProp = root.PropertyManager().GetPropertyFromPath( referenceAbsolutePath );

            if ( smart_ptr_isValid( targetProp ) )
            {
                targetProp->ConnectReference( smart_ptr<PropertyNode>( new PropertySelection( refData.Property ) ) );
                it = m_referencePathMap.erase( it );
            }
            else
            {
                it++;
            }
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ReferenceManager::LogUnresolvedReferences()
{
    std::map<std::string, sReferenceData>::iterator it;

    for ( it = m_referencePathMap.begin(); it != m_referencePathMap.end(); it++ )
    {
        sReferenceData refData = it->second;

        if ( (PropertyBase*)NULL != refData.Property )
        {
            ObjectNode* propertyParent = refData.Property->Parent();
            std::string propertyParentPath = "NULL";

            if ( NULL != propertyParent )
            {
                propertyParentPath = propertyParent->Path().PathString();
            }

            propertyParentPath += std::string(".") + refData.Property->Name();

            LOGGER( LOG_INFO << "Unresolved reference to: " << it->first << " from object: " << propertyParentPath );
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
    ObjectNode* propertyParent = prop->Parent();

    std::string retString = std::string( path );

    if ( NULL != propertyParent )
    {
        bool isDownHierarchy = (strncmp(retString.c_str(), "..", std::strlen("..")) == 0);
        bool isRelative = (strncmp(retString.c_str(), "/", std::strlen("/")) == 0);

        if ( isRelative || isDownHierarchy )
        {
            std::string propertyPath = propertyParent->Path().PathString();

            retString.erase(0, retString.find("/")+1);

            // We have to make path absolute
            if ( isDownHierarchy )
            {
                if ( propertyPath.rfind("/") )
                {
                    propertyPath.erase(propertyPath.rfind("/"), propertyPath.size());
                }
            }

            retString = propertyPath + std::string("/") + retString;
        }
    }

    return retString;
}

}
