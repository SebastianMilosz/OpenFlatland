#include "serializable.h"

#include <TextUtilities.h>
#include <LoggerUtilities.h>
#include <xmlformatter.h>
#include <iostream>
#include <exception>
#include <stdexcept>

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
    cSerializable::cSerializable( const std::string& name, cSerializable* parent ) :
        cSerializableStorage(),
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
    void cSerializable::SetName( std::string const& name )
    {
        m_sContainerName = name;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase* cSerializable::GetPropertyByName( std::string const& name )
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
      * @todo napisac implementacje
     **
    ******************************************************************************/
    cSerializable* cSerializable::ShareLevel(eShareLevel level )
    {
       m_shareLevel = level;
       return this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializable* cSerializable::LoadFromFile( std::string const& filePath, std::string const& container, bool createIfNotExist )
    {
        try
        {
            LOGGER( LOG_INFO  << ObjectName() << "-> LoadFromFile(" << filePath << ")" );

            if( createIfNotExist )
            {
                if( !utilities::file::IsFileExist( filePath ) )
                {
                    LOGGER( LOG_WARNING << "cSerializable::LoadFromFile: file: " << filePath << " does not exist."  );
                    SaveToFile( filePath, container );
                }
            }

            cXML          xml      ( filePath );
            cXmlFormatter formatter( dynamic_cast<cSerializableInterface*>(this), m_shareLevel );

            if( xml.Protocol() == "1.0" )
            {
                LOGGER( LOG_INFO  << "LoadFromFile v1.0" );
                formatter.LoadFromXML( xml.PointToNode( container ) );
            }
        }
        catch(exception &exc)
        {
            LOGGER( LOG_ERROR << ObjectName() <<  "-> LoadFromFile() exception: Type:" << typeid( exc ).name( ) << exc.what() );
        }
        catch (...)
        {
            LOGGER( LOG_ERROR << ObjectName() <<  "-> LoadFromFile() exception unknown");
        }

        return this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializable* cSerializable::SaveToFile( std::string const& filePath, std::string const& container )
    {
        try
        {
            LOGGER( LOG_INFO  << ObjectName() << "-> SaveToFile(" << filePath << ")" );

            cXML          xml;
            cXmlFormatter formatter( this, m_shareLevel );

            xml.PointToNode( container ).Add( formatter.SaveToXML() ).ToFile( filePath );
        }
        catch(exception &exc)
        {
            LOGGER( LOG_ERROR << ObjectName() << "-> SaveToFile() exception: Type:" << typeid( exc ).name( ) << exc.what() );
        }
        catch (...)
        {
            LOGGER( LOG_ERROR << ObjectName() << "-> SaveToFile() exception unknown" );
        }

        return this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializable* cSerializable::LoadFromXML( cXML xml, std::string const& container )
    {
        try
        {
            LOGGER( LOG_INFO  << ObjectName() << " -> LoadFromXML()" );

            cXmlFormatter formatter( this );

            formatter.LoadFromXML( xml.PointToNode( container ) );
        }
        catch(exception &exc)
        {
            LOGGER( LOG_ERROR << ObjectName() << "-> LoadFromXML() exception: Type:" << typeid( exc ).name( ) << exc.what() );
        }
        catch (...)
        {
            LOGGER( LOG_ERROR << ObjectName() << "-> LoadFromXML() exception unknown" );
        }

        return this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cXML cSerializable::SaveToXML( std::string const& container, int mode __attribute__((unused)) )
    {
        try
        {
            LOGGER( LOG_INFO << ObjectName() << "-> SaveToXML()" );

            cXmlFormatter formatter( this );

            return formatter.SaveToXML().PointToNode( container );
        }
        catch(exception &exc)
        {
            LOGGER( LOG_ERROR << ObjectName() << "-> SaveToXML() exception: Type:" << typeid( exc ).name( ) << exc.what() );
        }
        catch (...)
        {
            LOGGER( LOG_ERROR << ObjectName() << "-> SaveToXML() exception unknown" );
        }

        return cXML();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool cSerializable::IsPropertyUnique( std::string const& name ) const
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
    bool cSerializable::IsNameUnique( std::string const& name, bool checkParent ) const
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
      * @brief Return full object path
     **
    ******************************************************************************/
    std::string cSerializable::Path() const
    {
        std::string path;

        cSerializableInterface* parent = Parent();

        if( parent )
        {
           path = parent->Path() + "/" + path;
        }

        path += ObjectName();

        return path ;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string cSerializable::ObjectName() const
    {
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
    cSerializableInterface* cSerializable::GetChildByName( std::string const& name )
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
    cSerializableInterface* cSerializable::GetObjectFromPath( std::string const& path )
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
    PropertyBase* cSerializable::GetPropertyFromPath( std::string const& path )
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
