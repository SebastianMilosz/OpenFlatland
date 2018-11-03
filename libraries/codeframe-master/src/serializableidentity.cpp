#include "serializableidentity.hpp"
#include "serializableinterface.hpp"

#include <TextUtilities.h>
#include <LoggerUtilities.h>

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableIdentity::cSerializableIdentity( const std::string& name, cSerializableInterface& sint ) :
        m_Id( -1 ),
        m_sint( sint ),
        m_sContainerName( name )
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableIdentity::~cSerializableIdentity()
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializableIdentity::SetName( const std::string& name )
    {
        m_sContainerName = name;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string cSerializableIdentity::ObjectName( bool idSuffix ) const
    {
        if ( (GetId() >= 0) && (idSuffix == true) )
        {
            std::string cntName;
            cntName = m_sContainerName + std::string("[") + utilities::math::IntToStr( GetId() ) + std::string("]");
            return cntName;
        }
        return m_sContainerName;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    float cSerializableIdentity::LibraryVersion()
    {
        return 0.2;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string cSerializableIdentity::LibraryVersionString()
    {
        return std::string( "Serializable library version 0.2" );
    }
}
