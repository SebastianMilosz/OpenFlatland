#include "serializable.hpp"

#include <iostream>
#include <exception>
#include <stdexcept>

#include <TextUtilities.h>
#include <LoggerUtilities.h>
#include <xmlformatter.hpp>

using namespace std;

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializable::ParentUnbound()
    {
        m_parent->ChildList()->UnRegister( this );
        m_parent = NULL;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializable::ParentBound( cSerializableInterface* obj )
    {
        m_parent = obj;

        // Rejestrujemy sie u rodzica
        if( m_parent )
        {
            m_parent->ChildList()->Register( this );
        }

        if( m_delay )
        {
            //Start();
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializable::cSerializable( const std::string& name, cSerializableInterface* parent ) :
        cSerializableSelectable(),
        m_SerializablePath( *this ),
        m_SerializableStorage( *this ),
        m_delay( 0 ),
        m_parent( NULL ),
        m_sContainerName( name )

    {
        ParentBound( parent );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializable::PulseChanged( bool fullTree )
    {
        EnterPulseState();

        // Emitujemy sygnaly zmiany wszystkich propertisow
        for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if( temp )
            {
                temp->PulseChanged();
            }
        }

        LeavePulseState();

        if( fullTree )
        {
            // Zmuszamy dzieci do aktualizacji
            for( cSerializableChildList::iterator it = this->ChildList()->begin(); it != this->ChildList()->end(); ++it )
            {
                cSerializableInterface* iser = *it;

                if( iser )
                {
                    iser->PulseChanged( fullTree );
                }
                else
                {
                    throw std::runtime_error( "cSerializable::PulseChanged() cSerializable* iser = NULL" );
                }
            }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializable::SetName( const std::string& name )
    {
        m_sContainerName = name;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase* cSerializable::GetPropertyByName( const std::string& name )
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
    PropertyBase* cSerializable::GetPropertyById( uint32_t id )
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
    std::string cSerializable::GetNameById( uint32_t id ) const
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
    cSerializable::~cSerializable()
    {
        // Wyrejestrowywujemy sie u rodzica
        if( m_parent )
        {
            m_parent->ChildList()->UnRegister( this );
        }

        ClearPropertyList();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializable::RegisterProperty( PropertyBase* prop )
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
    void cSerializable::UnRegisterProperty( PropertyBase* prop )
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
    void cSerializable::ClearPropertyList()
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
    bool cSerializable::IsPropertyUnique( const std::string& name ) const
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
    bool cSerializable::IsNameUnique( const std::string& name, bool checkParent ) const
    {
        int octcnt = 0;

        // Jesli niema rodzica to jestesmy rootem wiec na tym poziomie jestesmy wyjatkowi
        if( this->Parent() == NULL )
        {
            return true;
        }

        // Rodzica sprawdzamy tylko na wyrazne zyczenie
        if( checkParent )
        {
            // Sprawdzamy czy rodzic jest wyjątkowy
            bool isParentUnique = this->Parent()->IsNameUnique( this->Parent()->ObjectName() );

            // Jesli rodzic nie jest wyjatkowy to dzieci tez nie są wiec niesprawdzamy dalej
            if( isParentUnique == false )
            {
                return false;
            }
        }

        // Jesli rodzic jest wyjatkowy sprawdzamy dzieci
        for( cSerializableChildList::iterator it = this->Parent()->ChildList()->begin(); it != this->Parent()->ChildList()->end(); ++it )
        {
            cSerializableInterface* iser = *it;

            if( iser )
            {
                if( iser->ObjectName() == name )
                {
                    octcnt++;
                }
            }
            else
            {
                throw std::runtime_error( "cSerializable::IsNameUnique() cSerializable* iser = NULL" );
            }
        }

        if(octcnt == 1 ) return true;
        else return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializablePath& cSerializable::Path()
    {
        return m_SerializablePath;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableStorage& cSerializable::Storage()
    {
        return m_SerializableStorage;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string cSerializable::ObjectName( bool idSuffix ) const
    {
        if( (GetId() >= 0) && (idSuffix == true) )
        {
            std::string cntName;

            cntName = m_sContainerName + utilities::math::IntToStr( GetId() );

            return cntName;
        }
        return m_sContainerName;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string cSerializable::SizeString() const
    {
        return utilities::math::IntToStr(size());
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableInterface* cSerializable::Parent() const
    {
        return m_parent;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableInterface* cSerializable::GetChildByName( const std::string& name )
    {
        for( cSerializableChildList::iterator it = ChildList()->begin(); it != ChildList()->end(); ++it )
        {
            cSerializableInterface* iser = *it;
            if( iser->ObjectName() == name ) return iser;
        }
        return NULL;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableInterface* cSerializable::GetRootObject()
    {
        if( Parent() ) return Parent()->GetRootObject();

        return this;
    }

    /*****************************************************************************/
    /**
      * @brief Return serializable object from string path
     **
    ******************************************************************************/
    cSerializableInterface* cSerializable::GetObjectFromPath( const std::string& path )
    {
        // Rozdzelamy stringa na kawalki
        vector<std::string>     tokens;
        std::string             delimiters = "/";
        cSerializableInterface* curObject = this;

        utilities::text::split( path, delimiters, tokens);

        // Sprawdzamy czy root sie zgadza
        if(tokens.size() == 0)
        {
            if( curObject->ObjectName() != path )
            {
                return NULL;
            }
        }
        else
        {
            std::string tempStr = tokens.at(0);
            if( curObject->ObjectName() != tempStr )
            {
                return NULL;
            }
        }

        // Po wszystkich skladnikach sciezki
        for( unsigned i = 1; i < tokens.size(); i++ )
        {
            std::string levelName = tokens.at(i);
            curObject = curObject->GetChildByName( levelName );
            if(curObject == NULL) break;
        }

        return curObject;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase* cSerializable::GetPropertyFromPath( const std::string& path )
    {
        // Wydzielamy sciezke od nazwy propertisa
        std::string::size_type found = path.find_last_of(".");
        std::string objPath      = path.substr( 0, found );
        std::string propertyName = path.substr( found+1  );

        cSerializableInterface* object = GetObjectFromPath( objPath );

        if( object )
        {
            PropertyBase* prop = object->GetPropertyByName( propertyName );
            return prop;
        }

        return NULL;
    }

    /*****************************************************************************/
    /**
      * @brief Zatwierdzenie wszystkich zmian obiektu i jego potomnych
     **
    ******************************************************************************/
    void cSerializable::CommitChanges()
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

        for( cSerializableChildList::iterator it = ChildList()->begin(); it != ChildList()->end(); ++it )
        {
            cSerializableInterface* iser = *it;
            if( iser )
            {
                iser->CommitChanges();
            }
        }

        Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializable::slotPropertyChangedGlobal( PropertyBase* prop )
    {
        signalPropertyChanged.Emit( prop );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializable::slotPropertyChanged( PropertyBase* prop __attribute__((unused)) )
    {
        #ifdef SERIALIZABLE_USE_WXWIDGETS
        wxUpdatePropertyValue( prop );
        #endif
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializable::Enable( bool val )
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

        for( cSerializableChildList::iterator it = ChildList()->begin(); it != ChildList()->end(); ++it )
        {
            cSerializableInterface* iser = *it;
            if( iser )
            {
                iser->Enable( val );
            }
        }

        Unlock();
    }

}
