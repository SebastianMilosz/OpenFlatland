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
            m_referencePathMap.insert( std::pair<std::string, PropertyBase*>(m_referencePath,m_property) );
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ReferenceManager::SetParent( PropertyBase* prop )
{
    if ( (m_referencePath.size() != 0) && (NULL != prop) && (NULL == m_property) )
    {
        m_property = prop;
        m_referencePathMap.insert( std::pair<std::string, PropertyBase*>(m_referencePath,m_property) );
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
            LOGGER( LOG_INFO << "Unresolved reference to: " << it->first << " from object: " << prop->Name() );
        }
        else
        {
            LOGGER( LOG_ERROR << "Unresolved reference to: " << it->first << " from object: NULL" );
        }
    }
}

}
