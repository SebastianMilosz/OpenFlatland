#include "cxml.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <exception>
#include <stdexcept>

#include <pugixml.hpp>

using namespace pugi;
using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cXML::Initialize(void)
{
    if( m_xmlDocument ) delete m_xmlDocument;
    m_xmlDocument = new pugi::xml_document();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXML::cXML() :
    m_valid(false),
    m_xmlDocument(NULL)
{
    Initialize();
    m_valid = true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXML::cXML( const std::string& filePath ) :
    m_valid(false),
    m_xmlDocument(NULL)
{
    Initialize();
    FromFile( filePath );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXML::cXML( const cXML& xml ) :
    m_valid(false),
    m_xmlDocument(NULL)
{
    Initialize();
    m_xmlDocument->reset( *xml.m_xmlDocument );
    m_valid = xml.m_valid;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXML::cXML( const char* data, int dsize ) :
    m_valid(false),
    m_xmlDocument(NULL)
{
    Initialize();
    FromBuffer( data, dsize );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXML::cXML( cXMLNode xmlNode ) :
    m_valid(false),
    m_xmlDocument(NULL)
{
    Initialize();
    m_xmlDocument->root().append_copy( *xmlNode.m_xmlNode );
    if( xmlNode.IsValid() == true ) m_valid = true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cXML::CreateXMLDeclaration( void )
{
    pugi::xml_node decl = m_xmlDocument->prepend_child( pugi::node_declaration );
    decl.append_attribute("version") = "1.0";
    decl.append_attribute("protocol") = "1.0";
    decl.append_attribute("encoding") = "UTF-8";
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string cXML::Protocol()
{
    pugi::xml_node decl = m_xmlDocument->child("xml");

    const char* protString = decl.attribute("protocol").value();

    if( protString ) { return protString; }
    return "0.0";
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXML::~cXML()
{
    if( m_xmlDocument ) delete m_xmlDocument;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXML& cXML::PointToNode( const std::string& name )
{
    (void)name;

    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXML& cXML::PointToRoot()
{

    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXML& cXML::Add( cXML& xml )
{
    m_xmlDocument->reset( *xml.m_xmlDocument );

    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string cXML::ToString()
{
    CreateXMLDeclaration();
    std::ostringstream stream;
    m_xmlDocument->save( stream );
    return stream.str();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cXML::ToFile( const std::string& filePath )
{
    CreateXMLDeclaration();
    std::ofstream filestream( filePath.c_str(), std::fstream::in | std::fstream::out | std::fstream::trunc );

    m_xmlDocument->save( filestream );
    filestream.close();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cXML::FromFile( const std::string& filePath )
{
    std::filebuf fb;
    if( fb.open(filePath.c_str(), std::ios::in) )
    {
        std::istream stream(&fb);
        xml_parse_result m_result = m_xmlDocument->load( stream, parse_declaration );
        fb.close();

        if( m_result ) m_valid = true;
        else m_valid = false;
    }
    else
    {
        throw std::runtime_error( "cXML::FromFile file does not exist" );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cXML::FromString( const std::string& xmlString )
{
    std::stringbuf sbuf( xmlString );
    std::istream stream( &sbuf );
    xml_parse_result m_result = m_xmlDocument->load( stream, parse_declaration );

    if( m_result ) m_valid = true;
    else m_valid = false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cXML::FromBuffer ( const char* data, int dsize )
{
    xml_parse_result m_result = m_xmlDocument->load( data, dsize );

    if( m_result ) m_valid = true;
    else m_valid = false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cXML::Dispose()
{
    m_xmlDocument->reset();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXML& cXML::operator=( cXML& arg ) // copy/move constructor is called to construct arg
{
    m_xmlDocument->reset( *arg.m_xmlDocument );
    m_valid = arg.m_valid;
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool cXML::IsValid() const
{
    return m_valid;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode cXML::GetChild( const std::string& name ) const
{
    return cXMLNode( m_xmlDocument->child( name.c_str() ) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode cXML::FindChildByAttribute(const char_t *name, const char_t* attr_name, const char_t* attr_value) const
{
    return cXMLNode( m_xmlDocument->find_child_by_attribute(name, attr_name, attr_value) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode cXML::Root()
{
    return cXMLNode( m_xmlDocument->root() );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode cXML::AppendChild( const char_t* name )
{
    return cXMLNode( m_xmlDocument->append_child( name ) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXMLNode cXML::FirstChild()
{
    return cXMLNode( m_xmlDocument->first_child() );
}
