#include "serializablepropertymanager.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase* cPropertyManager::GetPropertyByName( const std::string& name )
    {
        // Po wszystkich zarejestrowanych parametrach
        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp && temp->Name() == name )
            {
                return temp;
            }
        }

        LOGGER( LOG_FATAL << "cSerializable::GetPropertyByName(" << name << "): Out of range" );

        return &m_dummyProperty;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase* cPropertyManager::GetPropertyById( uint32_t id )
    {
        //m_Mutex.Lock();
        // Po wszystkic1h zarejestrowanych parametrach
        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp && temp->Id() == id )
            {
                return temp;
            }
        }
        //m_Mutex.Unlock();

        throw std::out_of_range( "cSerializable::GetPropertyById(" + utilities::math::IntToStr(id) + "): Out of range" );

        return NULL;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string cPropertyManager::GetNameById( uint32_t id ) const
    {
        std::string retName = "";

        Lock();
        // Po wszystkic1h zarejestrowanych parametrach
        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp && temp->Id() == id )
            {
                retName = temp->Name();
            }
        }
        Unlock();

        return retName;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyManager::PulseChanged()
    {
        // Emitujemy sygnaly zmiany wszystkich propertisow
        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp )
            {
                temp->PulseChanged();
            }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyManager::CommitChanges()
    {
        Lock();
        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp )
            {
                temp->CommitChanges();
            }
        }
        Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyManager::Enable( bool val )
    {
        // Po wszystkich propertisach ustawiamy nowy stan
        Lock();

        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp )
            {
                temp->Info().Enable( val );
            }
        }

        Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyManager::RegisterProperty( PropertyBase* prop )
    {
        Lock();
        m_vMainPropertyList.push_back( prop );
        prop->signalChanged.connect(this, &cSerializable::slotPropertyChanged       );
        prop->signalChanged.connect(this, &cSerializable::slotPropertyChangedGlobal );
        Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyManager::UnRegisterProperty( PropertyBase* prop )
    {
        Lock();
        // Po wszystkic1h zarejestrowanych parametrach
        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp && temp->Name() == prop->Name() )
            {
                // Wywalamy z listy
                temp->signalChanged.disconnect( this );
                m_vMainPropertyList.erase(m_vMainPropertyList.begin() + n);
                break;
            }
        }
        Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyManager::ClearPropertyList()
    {
        Lock();
        m_vMainPropertyList.clear();
        Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool cPropertyManager::IsPropertyUnique( const std::string& name ) const
    {
        int octcnt = 0;

        Lock();
        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp && temp->Name() == name )
            {
                octcnt++;
            }
        }
        Unlock();

        if(octcnt == 1 ) return true;
        else return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase* cPropertyManager::GetObjectFieldValue( int cnt )
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
    int cPropertyManager::GetObjectFieldCnt() const
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
    PropertyIterator cPropertyManager::begin() throw()
    {
        return PropertyIterator(this, 0);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyIterator cPropertyManager::end() throw()
    {
        return PropertyIterator(this, GetObjectFieldCnt());
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int cPropertyManager::size() const
    {
        return GetObjectFieldCnt();
    }
}
