#include "serializable_object_selection.hpp"

#include <assert.h>

namespace codeframe
{

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectSelection::ObjectSelection( ObjectNode* obj ) :
    m_selection( obj )
{
    assert( obj );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectSelection::~ObjectSelection()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectNode* ObjectSelection::GetNode( unsigned int id )
{
    return m_selection;
}

}
