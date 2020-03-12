#include "reference_manager.hpp"

#include <cstring>      // std::strlen

#include "serializable_object.hpp"
#include "serializable_property_selection.hpp"
#include "utilities/LoggerUtilities.h"

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
    m_property( nullptr )
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

    if (prop)
    {
        m_property = smart_ptr<PropertyNode>( new PropertySelection(prop) );
    }
    else
    {
        m_property = smart_ptr<PropertyNode>(nullptr);
    }

    if ( m_referencePath != "" )
    {
        if ( nullptr != m_property )
        {
            std::string referenceAbsolutePath( PreparePath( m_referencePath, m_property ) );
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
    if ( (m_referencePath != "") && (nullptr != prop) && (nullptr == m_property) )
    {
        m_property = smart_ptr<PropertyNode>( new PropertySelection(prop) );

        std::string referenceAbsolutePath( PreparePath( m_referencePath, m_property ) );
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
    for (auto it = m_referencePathMap.begin(); it != m_referencePathMap.end();)
    {
        sReferenceData refData = it->second;
        std::string path( it->first );

        std::string referenceAbsolutePath( PreparePath( refData.RefPath, refData.Property ) );

        if (smart_ptr_isValid(refData.Property))
        {
            smart_ptr<PropertyNode> targetProp = root.PropertyList().GetPropertyFromPath( referenceAbsolutePath );

            if (smart_ptr_isValid(targetProp))
            {
                refData.Property->ConnectReference(smart_ptr<PropertyNode>(targetProp));
                it = m_referencePathMap.erase(it);
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
    for (auto it = m_referencePathMap.begin(); it != m_referencePathMap.end(); it++)
    {
        sReferenceData refData( it->second );

        if (smart_ptr_isValid(refData.Property))
        {
            smart_ptr<ObjectNode> propertyParent(refData.Property->Parent());
            std::string propertyParentPath( "NULL" );

            if (nullptr != propertyParent)
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
std::string ReferenceManager::PreparePath( const std::string& path, smart_ptr<PropertyNode> prop )
{
    std::string retString( path );

    if (prop)
    {
        ObjectNode* propertyParent = prop->Parent();

        if (propertyParent)
        {
            bool isDownHierarchy = (strncmp(retString.c_str(), "..", std::strlen("..")) == 0);
            bool isRelative = (strncmp(retString.c_str(), "/", std::strlen("/")) == 0);

            if (isRelative || isDownHierarchy)
            {
                std::string propertyPath( propertyParent->Path().PathString() );

                retString.erase(0, retString.find("/")+1);

                // We have to make path absolute
                if (isDownHierarchy)
                {
                    if (propertyPath.rfind("/"))
                    {
                        propertyPath.erase(propertyPath.rfind("/"), propertyPath.size());
                    }
                }

                retString = propertyPath + std::string("/") + retString;
            }
        }
    }

    return retString;
}

}
