#include "serializable_object_multiple_selection.hpp"
#include "serializable_object.hpp"

#include <cassert>

namespace codeframe
{

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectMultipleSelection::ObjectMultipleSelection() :
    ObjectSelection()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectMultipleSelection::ObjectMultipleSelection( smart_ptr<ObjectNode> obj ) :
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
smart_ptr<ObjectNode> ObjectMultipleSelection::GetNode( unsigned int id )
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
std::string ObjectMultipleSelection::ObjectName( bool idSuffix ) const
{
    std::string retName;
    for(auto const& value: m_selection)
    {
        retName += value->Identity().ObjectName(idSuffix);
    }
    return retName;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ObjectMultipleSelection::Add( smart_ptr<ObjectNode> obj )
{
    m_selection.push_back( obj );
}

}
