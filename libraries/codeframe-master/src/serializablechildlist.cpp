#include "serializablechildlist.h"
#include "serializable.h"

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
    void cSerializableChildList::Register( cSerializable* child )
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
    void cSerializableChildList::UnRegister( cSerializable* child )
    {
        if( child == NULL ) return;

        m_childVector.erase(std::remove(m_childVector.begin(), m_childVector.end(), child), m_childVector.end());
        child->ParentUnbound();
        m_childCnt--;
    }
}
