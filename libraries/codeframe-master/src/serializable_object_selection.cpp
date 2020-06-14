#include "serializable_object_selection.hpp"
#include "serializable_object_dummy.hpp"
#include "serializable_property_base.hpp"
#include "serializable_property_selection.hpp"

#include <cassert>

namespace codeframe
{

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectSelection::ObjectSelection( smart_ptr<ObjectNode> obj ) :
    m_selection( obj )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectSelection::ObjectSelection() :
    m_selection( nullptr )
{

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
smart_ptr<PropertyNode> ObjectSelection::Property(const std::string& name)
{
    if ( smart_ptr_isValid(m_selection) )
    {
        return m_selection->Property(name);
    }

    static PropertyBase m_dummyProperty( nullptr, "DUMMY", TYPE_NON, cPropertyInfo() );
    return smart_ptr<PropertyNode>( new PropertySelection( &m_dummyProperty ));
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<PropertyNode> ObjectSelection::PropertyFromPath(const std::string& path)
{
    if ( smart_ptr_isValid(m_selection) )
    {
        return m_selection->PropertyFromPath(path);
    }

    static PropertyBase m_dummyProperty( nullptr, "DUMMY", TYPE_NON, cPropertyInfo() );
    return smart_ptr<PropertyNode>( new PropertySelection( &m_dummyProperty ));
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectNode> ObjectSelection::GetNode( unsigned int id )
{
    return m_selection;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
unsigned int ObjectSelection::GetNodeCount()
{
    if ( smart_ptr_isValid(m_selection) )
    {
        return 0U;
    }

    return 0U;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string ObjectSelection::ObjectName( bool idSuffix ) const
{
    if ( smart_ptr_isValid(m_selection) )
    {
        return m_selection->Identity().ObjectName(idSuffix);
    }

    return "";
}

}
