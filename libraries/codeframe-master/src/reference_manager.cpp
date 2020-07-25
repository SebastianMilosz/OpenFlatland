#include "reference_manager.hpp"

#include <cstring>      // std::strlen

#include "serializable_object.hpp"
#include "serializable_property_selection.hpp"
#include "utilities/LoggerUtilities.h"

namespace codeframe
{

std::map<PropertyNode*, ReferenceManager::sReferenceData> ReferenceManager::m_referencePathMap;

bool_t ReferenceManager::m_inhibitResolveReferences = false;

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
    if (m_inhibitResolveReferences == false)
    {
        for (auto it = m_referencePathMap.begin(); it != m_referencePathMap.end();)
        {
            const sReferenceData refData = it->second;

            if (refData.Property)
            {
                cPath::sPathLink pathLink;
                auto propertyParent = smart_ptr<ObjectSelection>(new ObjectSelection(refData.Property->Parent()));
                cPath::PreparePathLink(refData.RefPath, pathLink, propertyParent);

                if (smart_ptr_isValid(refData.Property))
                {
                    const std::string pathLinkString(pathLink.ToDirString());
                    smart_ptr<PropertyNode> targetProp = root.PropertyList().GetPropertyFromPath( pathLinkString );

                    if (smart_ptr_isValid(targetProp))
                    {
                        if ( refData.Property->ConnectReference(smart_ptr<PropertyNode>(targetProp)) )
                        {
                            it = m_referencePathMap.erase(it);
                        }
                    }
                    else
                    {
                        it++;
                    }
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
    for (const auto& iteam: m_referencePathMap)
    {
        const sReferenceData refData( iteam.second );

        if (smart_ptr_isValid(refData.Property))
        {
            const smart_ptr<ObjectNode> propertyParent(refData.Property->Parent());
            std::string propertyParentPath( "NULL" );

            if (nullptr != propertyParent)
            {
                propertyParentPath = propertyParent->Path().PathString();
            }

            propertyParentPath += std::string(".") + refData.Property->Name();

            LOGGER( LOG_INFO << "Unresolved reference to: " << refData.RefPath << " from object: " << propertyParentPath );
        }
        else
        {
            LOGGER( LOG_ERROR << "Unresolved reference to: " << refData.RefPath << " from object: NULL" );
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
unsigned int ReferenceManager::UnresolvedReferencesCount()
{
    return m_referencePathMap.size();
}

}
