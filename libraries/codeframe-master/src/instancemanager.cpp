#include "instancemanager.hpp"

namespace codeframe
{
    std::vector<void*>  cInstanceManager::s_vInstanceList;
    WrMutex             cInstanceManager::s_Mutex;
    int                 cInstanceManager::s_InstanceCnt;

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cInstanceManager::cInstanceManager()
    {
        s_Mutex.Lock();
        s_vInstanceList.push_back( (void*)this );
        s_InstanceCnt++;
        s_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cInstanceManager::~cInstanceManager()
    {
        s_Mutex.Lock();

        std::vector<void*>::iterator position = std::find( s_vInstanceList.begin(), s_vInstanceList.end(), (void*)this );

        // Jesli znaleziono to usuwamy
        if( position != s_vInstanceList.end() )
        {
            s_vInstanceList.erase( position );
            s_InstanceCnt--;
        }

        s_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief Zwraca prawde jesli obiekt jeszcze istnieje
     **
    ******************************************************************************/
    bool cInstanceManager::IsInstance( void* ptr )
    {
        bool retVal = false;
        s_Mutex.Lock();

        std::vector<void*>::iterator position = std::find( s_vInstanceList.begin(), s_vInstanceList.end(), (void*)ptr );

        // Jesli znaleziono to usuwamy
        if( position != s_vInstanceList.end() )
        {
            retVal = true;
        }

        s_Mutex.Unlock();

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cInstanceManager::DestructorEnter( void* ptr )
    {
        s_Mutex.Lock();

        std::vector<void*>::iterator position = std::find( s_vInstanceList.begin(), s_vInstanceList.end(), (void*)ptr );

        // Jesli znaleziono to usuwamy
        if( position != s_vInstanceList.end() )
        {
            s_vInstanceList.erase( position );
            s_InstanceCnt--;
        }

        s_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int cInstanceManager::GetInstanceCnt()
    {
        return s_InstanceCnt;
    }
}
