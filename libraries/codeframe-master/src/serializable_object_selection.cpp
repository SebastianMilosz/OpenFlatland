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
ObjectSelection::ObjectSelection( smart_ptr<ObjectNode> obj ) :
    m_smartSelection( obj ),
    m_selection( smart_ptr_getRaw(obj) )
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
ObjectSelection::operator ObjectNode&()
{
    if (m_selection)
    {
        return *m_selection;
    }

    static ObjectDummy dummyObject("dummyObject");
    return dummyObject;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<PropertyNode> ObjectSelection::Property(const std::string& name)
{
    if (m_selection)
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
    if (m_selection)
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
ObjectNode* ObjectSelection::GetNode( unsigned int id )
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
    return 0U;
}

}
