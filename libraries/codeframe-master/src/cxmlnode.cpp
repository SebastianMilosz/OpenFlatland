#include "cxmlnode.h"

#include <pugixml.hpp>

using namespace pugi;
using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cXMLNode::Initialize()
{
    if( m_xmlNode != NULL )
    {
        delete m_xmlNode;
    }
    m_xmlNode = new xml_node();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode::cXMLNode() : m_xmlNode(NULL)
{
    Initialize();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode::cXMLNode( const cXMLNode& node ) : m_xmlNode(NULL)
{
    Initialize();

    *m_xmlNode = *node.m_xmlNode;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode::cXMLNode( const pugi::xml_node& node ) : m_xmlNode(NULL)
{
    Initialize();

    *m_xmlNode = node;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode::~cXMLNode()
{
    if( m_xmlNode != NULL )
    {
        delete m_xmlNode;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode& cXMLNode::operator=(cXMLNode node)
{
    *this->m_xmlNode = *node.m_xmlNode;
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool cXMLNode::IsValid()
{
    if( *m_xmlNode ) return true;
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode cXMLNode::FindChildByAttribute(const char_t* name, const char_t* attr_name, const char_t* attr_value) const
{
    return cXMLNode( m_xmlNode->find_child_by_attribute( name, attr_name, attr_value ) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode cXMLNode::FindChildByAttribute(const char_t* attr_name, const char_t* attr_value) const
{
    return cXMLNode( m_xmlNode->find_child_by_attribute( attr_name, attr_value ) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode cXMLNode::Child(const char_t* name) const
{
    return cXMLNode( m_xmlNode->child( name ) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode cXMLNode::FirstChild( void ) const
{
    return cXMLNode( m_xmlNode->first_child() );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode cXMLNode::NextSibling( void ) const
{
    return cXMLNode( m_xmlNode->next_sibling() );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode cXMLNode::AppendChild(const char_t* name)
{
    return cXMLNode( m_xmlNode->append_child(name) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cXMLNode::AppendAttribute(const char_t* name, const char_t* value)
{
    m_xmlNode->append_attribute( name ) = value;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const char_t* cXMLNode::GetAttributeAsString(const char_t* name)
{
    return m_xmlNode->attribute(name).value();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int cXMLNode::GetAttributeAsInteger(const char_t* name)
{
    return m_xmlNode->attribute(name).as_int();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
double cXMLNode::GetAttributeAsDouble(const char_t* name)
{
    return m_xmlNode->attribute(name).as_double();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const char_t* cXMLNode::GetValueAsString( void )
{
    return m_xmlNode->child_value();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int cXMLNode::GetValueAsInteger( void )
{
    return int( m_xmlNode->text().as_int() );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cXMLNode::AppendCopy(const cXMLNode &node )
{
    m_xmlNode->append_copy( *node.m_xmlNode );
}
