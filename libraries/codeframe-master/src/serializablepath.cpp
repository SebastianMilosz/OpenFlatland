#include "serializablepath.hpp"

#include "serializableinterface.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializablePath::cSerializablePath( cSerializableInterface& sint ) :
        m_sint( sint )
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializablePath::~cSerializablePath()
    {

    }

    /*****************************************************************************/
    /**
      * @brief Return full object path
     **
    ******************************************************************************/
    std::string cSerializablePath::PathString() const
    {
        std::string path;

        cSerializableInterface* parent = m_sint.Parent();

        if( (cSerializableInterface*)NULL != parent )
        {
           path = parent->Path().PathString() + "/" + path;
        }

        path += m_sint.ObjectName();

        return path ;
    }
}
