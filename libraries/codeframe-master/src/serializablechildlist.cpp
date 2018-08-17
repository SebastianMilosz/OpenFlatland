#include "serializablechildlist.hpp"
#include "serializable.hpp"

#include <cstdio> // std::remove

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableChildList::cSerializableChildList() :
        m_childCnt( 0 )
    {
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializableChildList::Register( cSerializableInterface* child )
    {
        if( child )
        {
            m_childVector.push_back( child );
            m_childCnt++;
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializableChildList::UnRegister( cSerializableInterface* child )
    {
        if( child )
        {
            m_childVector.erase(std::remove(m_childVector.begin(), m_childVector.end(), child), m_childVector.end());
            m_childCnt--;
        }
    }
}
