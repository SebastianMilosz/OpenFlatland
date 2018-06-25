#include "serializableinterface.h"
#include "serializableproperty.h"

#include <LoggerUtilities.h>

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableInterface::cSerializableInterface() : m_dummyProperty(NULL, "DUMMY", TYPE_NON, cPropertyInfo())
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableInterface::~cSerializableInterface()
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase* cSerializableInterface::GetObjectFieldValue( int cnt )
    {
        m_Mutex.Lock();
        PropertyBase* retParameter = m_vMainPropertyList.at( cnt );
        m_Mutex.Unlock();

        return retParameter;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int cSerializableInterface::GetObjectFieldCnt() const
    {
        m_Mutex.Lock();
        int retSize = m_vMainPropertyList.size();
        m_Mutex.Unlock();

        return retSize;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableInterface::iterator cSerializableInterface::begin() throw()
    {
        return cSerializableInterface::iterator(this, 0);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableInterface::iterator cSerializableInterface::end() throw()
    {
        return cSerializableInterface::iterator(this, GetObjectFieldCnt());
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int cSerializableInterface::size() const
    {
        return GetObjectFieldCnt();
    }

}
