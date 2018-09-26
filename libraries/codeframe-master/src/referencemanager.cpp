#include "referencemanager.hpp"

#include <serializable.hpp>
#include <utilities/LoggerUtilities.h>

namespace codeframe
{

std::map<std::string, cSerializableInterface*> ReferenceManager::m_referencePathMap;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ReferenceManager::ReferenceManager() :
    m_referencePath(""),
    m_parent( NULL )
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
void ReferenceManager::SetReference( const std::string& refPath, cSerializableInterface* obj )
{
    m_referencePath = refPath;
    m_parent = obj;

    if ( m_referencePath.size() != 0 )
    {
        if ( NULL != m_parent )
        {
            m_referencePathMap.insert( std::pair<std::string, cSerializableInterface*>(m_referencePath,m_parent) );
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ReferenceManager::SetParent( cSerializableInterface* obj )
{
    if ( (m_referencePath.size() != 0) && (NULL != obj) && (NULL == m_parent) )
    {
        m_parent = obj;
        m_referencePathMap.insert( std::pair<std::string, cSerializableInterface*>(m_referencePath,m_parent) );
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
    std::map<std::string, cSerializableInterface*>::iterator it;

    for ( it = m_referencePathMap.begin(); it != m_referencePathMap.end(); it++ )
    {
        cSerializableInterface* obj = it->second;
        if ( (cSerializableInterface*)NULL != obj )
        {
            LOGGER( LOG_INFO << "Unresolved reference to: " << it->first << " from object: " << obj->ObjectName() );
        }
        else
        {
            LOGGER( LOG_ERROR << "Unresolved reference to: " << it->first << " from object: NULL" );
        }
    }
}

}
