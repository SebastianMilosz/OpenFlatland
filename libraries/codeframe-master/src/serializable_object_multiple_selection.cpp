#include "serializable_object_multiple_selection.hpp"

#include <cassert>

namespace codeframe
{

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectMultipleSelection::ObjectMultipleSelection( ObjectNode* obj ) :
    ObjectSelection()
{
    assert( obj );

    m_selection.push_back( obj );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectMultipleSelection::~ObjectMultipleSelection()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectNode* ObjectMultipleSelection::GetNode( unsigned int id )
{
    return m_selection.at( id );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
unsigned int ObjectMultipleSelection::GetNodeCount()
{
    return m_selection.size();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ObjectMultipleSelection::Add( ObjectNode* obj )
{
    m_selection.push_back( obj );
}

}
