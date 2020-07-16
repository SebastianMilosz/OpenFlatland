#include "serializable_object_selection.hpp"
#include "serializable_object_dummy.hpp"
#include "serializable_property_base.hpp"
#include "serializable_property_selection.hpp"

#include <cassert>
#include <LoggerUtilities.h>

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
    if (smart_ptr_isValid(obj)==false)
    {
        assert(true);
    }
    obj->signalDeleted.connect(this, &ObjectSelection::OnDelete);
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
        return 1U;
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

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string ObjectSelection::PathString() const
{
    if ( smart_ptr_isValid(m_selection) )
    {
        return m_selection->Path().PathString();
    }

    return "";
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectSelection::Parent() const
{
    if ( smart_ptr_isValid(m_selection) )
    {
        return m_selection->Path().Parent();
    }

    return smart_ptr<ObjectSelection>(nullptr);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectSelection::Root()
{
    if ( smart_ptr_isValid(m_selection) )
    {
        return m_selection->Path().GetRootObject();
    }

    return smart_ptr<ObjectSelection>(nullptr);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectSelection::ObjectFromPath( const std::string& path )
{
    if ( smart_ptr_isValid(m_selection) )
    {
        return m_selection->Path().GetObjectFromPath(path);
    }

    return smart_ptr<ObjectSelection>(nullptr);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectSelection::GetObjectByName( const std::string& name )
{
    if ( smart_ptr_isValid(m_selection) )
    {
        return m_selection->ChildList().GetObjectByName(name);
    }

    return smart_ptr<ObjectSelection>(nullptr);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectSelection::GetObjectById( const uint32_t id )
{
    if ( smart_ptr_isValid(m_selection) )
    {
        return m_selection->Child(id);
    }

    return smart_ptr<ObjectSelection>(nullptr);
}

/*****************************************************************************/
/**
  * @brief This method should return true if all objects in selection exist
 **
******************************************************************************/
bool_t ObjectSelection::IsValid() const
{
    if ( smart_ptr_isValid(m_selection) )
    {
        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ObjectSelection::OnDelete(void* deletedPtr)
{
    m_selection = smart_ptr<ObjectNode>(nullptr);
}

}
