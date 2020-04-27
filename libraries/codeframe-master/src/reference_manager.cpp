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
        m_property = smart_ptr<PropertyNode>(new PropertySelection(prop));

        if (m_referencePath != "")
        {
            sReferenceData refData;
            refData.Property = m_property;
            refData.RefPath = m_referencePath;
            m_referencePathMap.insert( std::pair<PropertyNode*, sReferenceData>(prop, refData) );

#ifdef CODE_FRAME_REFERENCE_MANAGER_DEBUG
            LOGGER( LOG_INFO << "ReferenceManager::SetReference( const std::string& refPath, PropertyBase* prop ): " << utilities::math::PointerToHex((void*)prop) << " m_referencePath: " << m_referencePath << " PropPath: " << m_property->Path() );
#endif // CODE_FRAME_REFERENCE_MANAGER_DEBUG
        }
    }
    else
    {
        m_property = smart_ptr<PropertyNode>(nullptr);
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ReferenceManager::SetProperty( PropertyBase* prop )
{
    if (prop)
    {
        m_property = smart_ptr<PropertyNode>( new PropertySelection(prop) );

        if (m_referencePath != "")
        {
            sReferenceData refData;
            refData.Property = m_property;
            refData.RefPath = m_referencePath;
            m_referencePathMap.insert( std::pair<PropertyNode*, sReferenceData>(prop, refData) );

#ifdef CODE_FRAME_REFERENCE_MANAGER_DEBUG
            LOGGER( LOG_INFO << "ReferenceManager::SetProperty( PropertyBase* prop ): " << utilities::math::PointerToHex((void*)prop) << " m_referencePath: " << m_referencePath << " PropPath: " << m_property->Path());
#endif // CODE_FRAME_REFERENCE_MANAGER_DEBUG
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
#ifdef CODE_FRAME_REFERENCE_MANAGER_DEBUG
    LOGGER( LOG_INFO << "ResolveReferences for root object: " << root.Path().PathString() );
#endif // CODE_FRAME_REFERENCE_MANAGER_DEBUG

    for (auto it = m_referencePathMap.begin(); it != m_referencePathMap.end();)
    {
        sReferenceData refData = it->second;

        if (refData.Property)
        {
            std::vector<cPath::sPathNode> pathDir;
            auto propertyParent = smart_ptr<ObjectSelection>(new ObjectSelection(refData.Property->Parent()));
            cPath::PreparePath(refData.RefPath, pathDir, propertyParent);

#ifdef CODE_FRAME_REFERENCE_MANAGER_DEBUG
            LOGGER( LOG_INFO << "ResolveReferences: AbsolutePath=" << pathDir );
#endif // CODE_FRAME_REFERENCE_MANAGER_DEBUG

            if (smart_ptr_isValid(refData.Property))
            {
                smart_ptr<PropertyNode> targetProp = root.PropertyList().GetPropertyFromPath( pathDir );

                if (smart_ptr_isValid(targetProp))
                {
                    refData.Property->ConnectReference(smart_ptr<PropertyNode>(targetProp));
                    it = m_referencePathMap.erase(it);
#ifdef CODE_FRAME_REFERENCE_MANAGER_DEBUG
                    LOGGER( LOG_INFO << "ResolveReferences: REFERENCE CONNECTED!!!" );
#endif // CODE_FRAME_REFERENCE_MANAGER_DEBUG
                }
                else
                {
                    it++;
                }
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

}
