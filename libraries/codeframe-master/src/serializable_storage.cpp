#include "serializable_storage.hpp"

#include <iostream>
#include <exception>
#include <stdexcept>
#include <LoggerUtilities.h>

#include "serializable_object_node.hpp"
#include "reference_manager.hpp"

namespace codeframe
{

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cStorage::cStorage( ObjectNode& sint ) :
    m_shareLevel( ShareFull ),
    m_sint( sint )
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cStorage::~cStorage()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectNode& cStorage::ShareLevel( eShareLevel level )
{
   m_shareLevel = level;
   return m_sint;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectNode& cStorage::LoadFromFile( const std::string& filePath, const std::string& container, bool createIfNotExist )
{
    try
    {
        LOGGER( LOG_INFO  << m_sint.Identity().ObjectName() << "-> LoadFromFile(" << filePath << ")" );

        // We temporary suspend reference resolving
        ReferenceManager::Inhibit referenceManagerLock(m_sint);

        if ( createIfNotExist )
        {
            if ( !utilities::file::IsFileExist( filePath ) )
            {
                LOGGER( LOG_WARNING << "cSerializable::LoadFromFile: file: " << filePath << " does not exist."  );
                SaveToFile( filePath, container );
            }
        }

        cXML          xml      ( filePath );
        cXmlFormatter formatter( m_sint, m_shareLevel );

        if ( xml.Protocol() == "1.0" )
        {
            LOGGER( LOG_INFO  << "LoadFromFile v1.0" );
            formatter.LoadFromXML( xml.PointToNode( container ) );
            m_sint.PulseChanged( true );
        }
    }
    catch ( std::exception& exc )
    {
        LOGGER( LOG_ERROR << m_sint.Identity().ObjectName() <<  "-> LoadFromFile() exception: Type:" << typeid( exc ).name( ) << exc.what() );
    }
    catch (...)
    {
        LOGGER( LOG_ERROR << m_sint.Identity().ObjectName() <<  "-> LoadFromFile() exception unknown");
    }

    return m_sint;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectNode& cStorage::SaveToFile( const std::string& filePath, const std::string& container )
{
    try
    {
        LOGGER( LOG_INFO  << m_sint.Identity().ObjectName() << "-> SaveToFile(" << filePath << ")" );

        cXML          xml;
        cXmlFormatter formatter( m_sint, m_shareLevel );

        cXML formXml = formatter.SaveToXML();

        xml.PointToNode( container ).Add( formXml ).ToFile( filePath );
    }
    catch ( std::exception& exc )
    {
        LOGGER( LOG_ERROR << m_sint.Identity().ObjectName() << "-> SaveToFile() exception: Type:" << typeid( exc ).name( ) << exc.what() );
    }
    catch (...)
    {
        LOGGER( LOG_ERROR << m_sint.Identity().ObjectName() << "-> SaveToFile() exception unknown" );
    }

    return m_sint;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectNode& cStorage::LoadFromXML( cXML xml, const std::string& container )
{
    try
    {
        LOGGER( LOG_INFO  << m_sint.Identity().ObjectName() << " -> LoadFromXML()" );

        cXmlFormatter formatter( m_sint );

        formatter.LoadFromXML( xml.PointToNode( container ) );
    }
    catch ( std::exception& exc )
    {
        LOGGER( LOG_ERROR << m_sint.Identity().ObjectName() << "-> LoadFromXML() exception: Type:" << typeid( exc ).name( ) << exc.what() );
    }
    catch (...)
    {
        LOGGER( LOG_ERROR << m_sint.Identity().ObjectName() << "-> LoadFromXML() exception unknown" );
    }

    return m_sint;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cXML cStorage::SaveToXML( const std::string& container, int mode __attribute__((unused)) )
{
    try
    {
        LOGGER( LOG_INFO << m_sint.Identity().ObjectName() << "-> SaveToXML()" );

        cXmlFormatter formatter( m_sint );

        return formatter.SaveToXML().PointToNode( container );
    }
    catch ( std::exception& exc )
    {
        LOGGER( LOG_ERROR << m_sint.Identity().ObjectName() << "-> SaveToXML() exception: Type:" << typeid( exc ).name( ) << exc.what() );
    }
    catch (...)
    {
        LOGGER( LOG_ERROR << m_sint.Identity().ObjectName() << "-> SaveToXML() exception unknown" );
    }

    return cXML();
}

}
