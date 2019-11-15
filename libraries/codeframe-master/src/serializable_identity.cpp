#include "serializable_identity.hpp"
#include "serializable_object_node.hpp"

#include <TextUtilities.h>
#include <LoggerUtilities.h>

namespace codeframe
{
    uint32_t cIdentity::g_uid = 0U;

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cIdentity::cIdentity( const std::string& name, ObjectNode& sint ) :
        m_id( -1 ),
        m_uid( g_uid++ ),
        m_sint( sint ),
        m_sContainerName( name )
    {
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cIdentity::~cIdentity()
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cIdentity::SetName( const std::string& name )
    {
        m_sContainerName = name;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string cIdentity::ObjectName( bool idSuffix ) const
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
    float cIdentity::LibraryVersion()
    {
        return 0.2;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string cIdentity::LibraryVersionString()
    {
        return std::string( "codeframe library version 0.2" );
    }
}
