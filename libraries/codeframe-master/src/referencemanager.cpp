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
    m_referencePath("")
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
void ReferenceManager::Set( const std::string& refPath, cSerializableInterface* obj )
{
    m_referencePath = refPath;
    m_referencePathMap.insert( std::pair<std::string, cSerializableInterface*>(m_referencePath,obj) );
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
            std::string refPath = it->first;
            LOGGER( LOG_INFO << "Unresolved reference to: " << refPath << " from object: " << obj->ObjectName() );
        }
    }
}

}
