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
smart_ptr<PropertyNode> ObjectMultipleSelection::Property(const std::string& name)
{
    for (auto const& value: m_selection)
    {
        smart_ptr<PropertyNode> retObject = value->Property(name);
        if (smart_ptr_isValid(retObject))
        {
            return retObject;
        }
    }
    return smart_ptr<PropertyNode>(nullptr);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<PropertyNode> ObjectMultipleSelection::PropertyFromPath(const std::string& path)
{
    for (auto const& value: m_selection)
    {
        smart_ptr<PropertyNode> retObject = value->PropertyFromPath(path);
        if (smart_ptr_isValid(retObject))
        {
            return retObject;
        }
    }
    return smart_ptr<PropertyNode>(nullptr);
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
        retName += value->ObjectName(idSuffix);
    }
    return retName;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string ObjectMultipleSelection::PathString() const
{
    std::string retName;
    for(auto const& value: m_selection)
    {
        retName += value->Path().PathString();
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

/*****************************************************************************/
/**
  * @brief Return parent of object inside selection
  * @note selection can be taken only for objects on the same level of the path
  * thats why parent from any object within selection will be correct
  * parent of the selection
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectMultipleSelection::Parent() const
{
    if (m_selection.size() > 0U)
    {
        return m_selection.at(0U)->Path().Parent();
    }
    return smart_ptr<ObjectSelection>(nullptr);
}

/*****************************************************************************/
/**
  * @brief Return root node of this object
  * @note assumption ware taken that all selections must be on the same root node
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectMultipleSelection::Root()
{
    if (m_selection.size() > 0U)
    {
        return m_selection.at(0U)->Path().GetRootObject();
    }
    return smart_ptr<ObjectSelection>(nullptr);
}

/*****************************************************************************/
/**
  * @brief Return first object from selection with specific path
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectMultipleSelection::ObjectFromPath( const std::string& path )
{
    for (auto const& value: m_selection)
    {
        smart_ptr<ObjectSelection> retObject = value->Path().GetObjectFromPath(path);
        if (smart_ptr_isValid(retObject))
        {
            return retObject;
        }
    }
    return smart_ptr<ObjectSelection>(nullptr);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectMultipleSelection::GetObjectByName( const std::string& name )
{
    for (auto const& value: m_selection)
    {
        smart_ptr<ObjectSelection> retObject = value->ChildList().GetObjectByName(name);
        if (smart_ptr_isValid(retObject))
        {
            return retObject;
        }
    }
    return smart_ptr<ObjectSelection>(nullptr);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectMultipleSelection::GetObjectById( const uint32_t id )
{
    for (auto const& value: m_selection)
    {
        smart_ptr<ObjectSelection> retObject = value->ChildList().GetObjectById(id);
        if (smart_ptr_isValid(retObject))
        {
            return retObject;
        }
    }
    return smart_ptr<ObjectSelection>(nullptr);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ObjectMultipleSelection::OnDelete(void* deletedPtr)
{
    for (auto& value: m_selection)
    {
        if (smart_ptr_getRaw(value) == deletedPtr)
        {
            value = smart_ptr<ObjectNode>(nullptr);
            return;
        }
    }
}

}
