#include "reference_manager.hpp"

#include <cstring>      // std::strlen

#include "serializable_object.hpp"
#include "serializable_property_selection.hpp"
#include "utilities/LoggerUtilities.h"

namespace codeframe
{

std::map<PropertyNode*, ReferenceManager::sReferenceData> ReferenceManager::m_referencePathMap;

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
            auto propertyParent = smart_ptr<ObjectSelection>(new ObjectSelection(m_property->Parent()));
            std::string referenceAbsolutePath( PreparePath( m_referencePath, propertyParent ) );
            sReferenceData refData;
            refData.Property = m_property;
            refData.RefPath = m_referencePath;
            m_referencePathMap.insert( std::pair<PropertyNode*, sReferenceData>(prop, refData) );

            LOGGER( LOG_INFO << "ReferenceManager::SetReference( const std::string& refPath, PropertyBase* prop ): " << utilities::math::PointerToHex((void*)prop) << " AbsolutePath: " << referenceAbsolutePath << " m_referencePath: " << m_referencePath << " PropPath: " << m_property->Path() );
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

        if (m_property)
        {
            auto propertyParent = smart_ptr<ObjectSelection>(new ObjectSelection(m_property->Parent()));
            std::string referenceAbsolutePath( PreparePath( m_referencePath, propertyParent ) );
            sReferenceData refData;
            refData.Property = m_property;
            refData.RefPath = m_referencePath;

            m_referencePathMap.insert( std::pair<PropertyNode*, sReferenceData>(prop, refData) );

            LOGGER( LOG_INFO << "ReferenceManager::SetProperty( PropertyBase* prop ): " << utilities::math::PointerToHex((void*)prop) << " AbsolutePath: " << referenceAbsolutePath << " m_referencePath: " << m_referencePath << " PropPath: " << m_property->Path());
        }
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
    LOGGER( LOG_INFO << "ResolveReferences:BEG m_referencePathMap.size()=" << m_referencePathMap.size() );

    for (auto it = m_referencePathMap.begin(); it != m_referencePathMap.end();)
    {
        sReferenceData refData = it->second;

        if (refData.Property)
        {
            LOGGER( LOG_INFO << "ResolveReferences: refData.RefPath=" << refData.RefPath );

            auto propertyParent = smart_ptr<ObjectSelection>(new ObjectSelection(refData.Property->Parent()));
            std::string referenceAbsolutePath( PreparePath( refData.RefPath, propertyParent ) );

            if (smart_ptr_isValid(refData.Property))
            {
                smart_ptr<PropertyNode> targetProp = root.PropertyList().GetPropertyFromPath( referenceAbsolutePath );

                if (smart_ptr_isValid(targetProp))
                {
                    refData.Property->ConnectReference(smart_ptr<PropertyNode>(targetProp));
                    LOGGER( LOG_INFO << "ResolveReferences:REFERENCE CONNECTED!!!" );
                    it = m_referencePathMap.erase(it);
                }
                else
                {
                    it++;
                }
            }
        }
    }

    LOGGER( LOG_INFO << "ResolveReferences:END m_referencePathMap.size()=" << m_referencePathMap.size() );
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
                LOGGER( LOG_INFO << "LogUnresolvedReferences PathString: " << propertyParentPath );
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
std::string ReferenceManager::PreparePath( const std::string& path, smart_ptr<ObjectSelection> propertyParent )
{
    std::string retString( path );

    if (propertyParent)
    {
        bool isDownHierarchy = (strncmp(retString.c_str(), "..", std::strlen("..")) == 0);
        bool isRelative = (strncmp(retString.c_str(), "/", std::strlen("/")) == 0);

        if (isRelative || isDownHierarchy)
        {
            std::string propertyPath( propertyParent->GetNode()->Path().PathString() );

            retString.erase(0, retString.find("/")+1);

            // We have to make path absolute
            if (isDownHierarchy)
            {
                if (IsDownHierarchy(retString))
                {
                    smart_ptr<ObjectSelection> parentObject = propertyParent->GetNode()->Path().Parent();
                    retString = PreparePath(retString, parentObject );
                }

                if (propertyPath.rfind("/")!=std::string::npos)
                {
                    propertyPath.erase(propertyPath.rfind("/"), propertyPath.size());
                }
            }

            retString = propertyPath + std::string("/") + retString;
        }
    }

    return retString;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool_t ReferenceManager::IsDownHierarchy(const std::string& path)
{
    return (strncmp(path.c_str(), "..", std::strlen("..")) == 0);
}

}
