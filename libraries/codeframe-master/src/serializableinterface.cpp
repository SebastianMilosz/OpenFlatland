#include "serializableinterface.hpp"
#include "serializableproperty.hpp"

#include <LoggerUtilities.h>

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableInterface::cSerializableInterface() :
        m_dummyProperty(NULL, "DUMMY", TYPE_NON, cPropertyInfo()),
        m_Id( -1 )
    {
        static TypeInitializer typeInitializer;
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
    PropertyIterator cSerializableInterface::begin() throw()
    {
        return PropertyIterator(this, 0);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyIterator cSerializableInterface::end() throw()
    {
        return PropertyIterator(this, GetObjectFieldCnt());
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
