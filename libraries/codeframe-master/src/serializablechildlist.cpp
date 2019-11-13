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
    void cSerializableChildList::Register( ObjectNode* child )
    {
        if( child )
        {
            m_Mutex.Lock();
            m_childVector.push_back( child );
            m_childCnt++;
            m_Mutex.Unlock();
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializableChildList::UnRegister( ObjectNode* child )
    {
        if( child )
        {
            m_Mutex.Lock();
            m_childVector.erase(std::remove(m_childVector.begin(), m_childVector.end(), child), m_childVector.end());
            m_childCnt--;
            m_Mutex.Unlock();
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializableChildList::PulseChanged( bool fullTree )
    {
        m_Mutex.Lock();
        // Zmuszamy dzieci do aktualizacji
        for ( cSerializableChildList::iterator it = begin(); it != end(); ++it )
        {
            ObjectNode* iser = *it;

            if ( iser )
            {
                iser->PulseChanged( fullTree );
            }
        }
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief Zatwierdzenie wszystkich zmian obiektu i jego potomnych
     **
    ******************************************************************************/
    void cSerializableChildList::CommitChanges()
    {
        m_Mutex.Lock();
        for( cSerializableChildList::iterator it = begin(); it != end(); ++it )
        {
            ObjectNode* iser = *it;
            if( iser )
            {
                iser->CommitChanges();
            }
        }
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializableChildList::Enable( bool val )
    {
        m_Mutex.Lock();
        for( cSerializableChildList::iterator it = begin(); it != end(); ++it )
        {
            ObjectNode* iser = *it;
            if( iser )
            {
                iser->Enable( val );
            }
        }
        m_Mutex.Unlock();
    }
}
