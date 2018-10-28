#include "serializablestorage.hpp"

#include <iostream>
#include <exception>
#include <stdexcept>
#include <LoggerUtilities.h>

#include "serializableinterface.hpp"
#include "referencemanager.hpp"

namespace codeframe
{

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSerializableStorage::cSerializableStorage( cSerializableInterface& sint ) :
    m_shareLevel( ShareFull ),
    m_sint( sint )
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSerializableStorage::~cSerializableStorage()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSerializableInterface& cSerializableStorage::ShareLevel( eShareLevel level )
{
   m_shareLevel = level;
   return m_sint;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSerializableInterface& cSerializableStorage::LoadFromFile( const std::string& filePath, const std::string& container, bool createIfNotExist )
{
    try
    {
        LOGGER( LOG_INFO  << m_sint.Identity().ObjectName() << "-> LoadFromFile(" << filePath << ")" );

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
            ReferenceManager::ResolveReferences( m_sint );
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
cSerializableInterface& cSerializableStorage::SaveToFile( const std::string& filePath, const std::string& container )
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
cSerializableInterface& cSerializableStorage::LoadFromXML( cXML xml, const std::string& container )
{
    try
    {
        LOGGER( LOG_INFO  << m_sint.Identity().ObjectName() << " -> LoadFromXML()" );

        cXmlFormatter formatter( m_sint );

        formatter.LoadFromXML( xml.PointToNode( container ) );

        ReferenceManager::ResolveReferences( m_sint );
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
cXML cSerializableStorage::SaveToXML( const std::string& container, int mode __attribute__((unused)) )
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
