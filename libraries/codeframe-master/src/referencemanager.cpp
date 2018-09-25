#include "referencemanager.hpp"

namespace codeframe
{

std::list<std::string> ReferenceManager::m_referencePathList;

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
void ReferenceManager::Set( const std::string& refPath )
{
    m_referencePath = refPath;
    m_referencePathList.push_back( refPath );
    m_referencePathList.unique();
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

}
