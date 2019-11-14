#include "fontfactory.hpp"

bool     FontFactory::m_initialized = false;
sf::Font FontFactory::m_font;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
FontFactory::FontFactory( const std::string& name, ObjectNode* parent ) :
    Object( name, parent )
{
    if( m_initialized == false )
    {
        // Load it from a file
        if ( m_font.loadFromFile( "arial.ttf" ) )
        {
            m_initialized = true;
        }
        else
        {

        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
FontFactory::~FontFactory()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
sf::Font& FontFactory::GetFont()
{
    return m_font;
}
