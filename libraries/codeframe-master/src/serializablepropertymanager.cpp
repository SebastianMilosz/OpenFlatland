#include "serializablepropertymanager.hpp"

#include <LoggerUtilities.h>

#include "serializableinterface.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cPropertyManager::cPropertyManager( cSerializableInterface& sint ) :
        m_sint( sint ),
        m_dummyProperty(NULL, "DUMMY", TYPE_NON, cPropertyInfo())
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cPropertyManager::~cPropertyManager()
    {

    }

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
    PropertyBase* cPropertyManager::GetPropertyFromPath( const std::string& path )
    {
        // Wydzielamy sciezke od nazwy propertisa
        std::string::size_type found = path.find_last_of(".");
        std::string objPath      = path.substr( 0, found );
        std::string propertyName = path.substr( found+1  );

        cSerializableInterface* object = m_sint.Path().GetObjectFromPath( objPath );

        if( object )
        {
            PropertyBase* prop = object->PropertyManager().GetPropertyByName( propertyName );
            return prop;
        }

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

        m_sint.Lock();
        // Po wszystkic1h zarejestrowanych parametrach
        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp && temp->Id() == id )
            {
                retName = temp->Name();
            }
        }
        m_sint.Unlock();

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
        m_sint.Lock();
        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp )
            {
                temp->CommitChanges();
            }
        }
        m_sint.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyManager::Enable( bool val )
    {
        // Po wszystkich propertisach ustawiamy nowy stan
        m_sint.Lock();
        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp )
            {
                temp->Info().Enable( val );
            }
        }
        m_sint.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyManager::RegisterProperty( PropertyBase* prop )
    {
        m_sint.Lock();
        m_vMainPropertyList.push_back( prop );
        prop->signalChanged.connect(this, &cPropertyManager::slotPropertyChanged       );
        prop->signalChanged.connect(this, &cPropertyManager::slotPropertyChangedGlobal );
        m_sint.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyManager::UnRegisterProperty( PropertyBase* prop )
    {
        m_sint.Lock();
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
        m_sint.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyManager::ClearPropertyList()
    {
        m_sint.Lock();
        m_vMainPropertyList.clear();
        m_sint.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool cPropertyManager::IsPropertyUnique( const std::string& name ) const
    {
        int octcnt = 0;

        m_sint.Lock();
        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp && temp->Name() == name )
            {
                octcnt++;
            }
        }
        m_sint.Unlock();

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
        m_sint.Lock();
        PropertyBase* retParameter = m_vMainPropertyList.at( cnt );
        m_sint.Unlock();

        return retParameter;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int cPropertyManager::GetObjectFieldCnt() const
    {
        m_sint.Lock();
        int retSize = m_vMainPropertyList.size();
        m_sint.Unlock();

        return retSize;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyIterator cPropertyManager::begin() throw()
    {
        return PropertyIterator( *this, 0 );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyIterator cPropertyManager::end() throw()
    {
        return PropertyIterator( *this, GetObjectFieldCnt() );
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

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyManager::slotPropertyChangedGlobal( PropertyBase* prop )
    {
        signalPropertyChanged.Emit( prop );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyManager::slotPropertyChanged( PropertyBase* prop )
    {
        #ifdef SERIALIZABLE_USE_WXWIDGETS
        wxUpdatePropertyValue( prop );
        #endif
    }
}
