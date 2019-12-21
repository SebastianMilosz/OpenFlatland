#include "serializable_property_list.hpp"
#include "serializable_property_selection.hpp"
#include "serializable_object_node.hpp"
#include "serializable_property_base.hpp"
#include "serializable_property_multiple_selection.hpp"

#include <LoggerUtilities.h>

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cPropertyList::cPropertyList( ObjectNode& sint ) :
        m_sint( sint ),
        m_dummyProperty( nullptr, "DUMMY", TYPE_NON, cPropertyInfo() )
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cPropertyList::~cPropertyList()
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<PropertyNode> cPropertyList::GetPropertyByName( const std::string& name )
    {
        // Po wszystkich zarejestrowanych parametrach
        for( auto temp : m_vMainPropertyList )
        {
            if( temp && temp->Name() == name )
            {
                return smart_ptr<PropertyNode>( new PropertySelection(temp) );
            }
        }

        LOGGER( LOG_FATAL << "cSerializable::GetPropertyByName(" << name << "): Out of range" );

        return smart_ptr<PropertyNode>( new PropertySelection( &m_dummyProperty ) );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<PropertyNode> cPropertyList::GetPropertyById( const uint32_t id )
    {
        //m_Mutex.Lock();
        // Po wszystkic1h zarejestrowanych parametrach
        for( auto temp : m_vMainPropertyList )
        {
            if( temp && temp->Id() == id )
            {
                return smart_ptr<PropertyNode>( new PropertySelection( temp ) );
            }
        }
        //m_Mutex.Unlock();

        throw std::out_of_range( "cSerializable::GetPropertyById(" + utilities::math::IntToStr(id) + "): Out of range" );

        return smart_ptr<PropertyNode>( nullptr );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<PropertyNode> cPropertyList::GetPropertyFromPath( const std::string& path )
    {
        // Wydzielamy sciezke od nazwy propertisa
        std::string::size_type found( path.find_last_of(".") );
        std::string objPath( path.substr( 0, found ) );
        std::string propertyName( path.substr( found+1 ) );

        smart_ptr<ObjectSelection> objectSelection = m_sint.Path().GetObjectFromPath( objPath );

        if ( smart_ptr_isValid( objectSelection ) )
        {
            if ( objectSelection->GetNodeCount() >= 1U )
            {
                smart_ptr<PropertyMultipleSelection> propMultiNode( new PropertyMultipleSelection() );

                for ( ObjectNode* obj : *objectSelection )
                {
                    smart_ptr<PropertyNode> node = obj->PropertyList().GetPropertyByName( propertyName );
                    if ( nullptr != node )
                    {
                        propMultiNode->Add( node );
                    }
                }
                return propMultiNode;
            }
            else
            {
                smart_ptr<PropertyNode> propNode = objectSelection->GetNode()->PropertyList().GetPropertyByName( propertyName );
                return propNode;
            }
        }

        return smart_ptr<PropertyNode>( nullptr );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string cPropertyList::GetNameById( uint32_t id ) const
    {
        std::string retName( "" );

        m_Mutex.Lock();
        // Po wszystkic1h zarejestrowanych parametrach
        for( auto temp : m_vMainPropertyList )
        {
            if ( temp && temp->Id() == id )
            {
                retName = temp->Name();
            }
        }
        m_Mutex.Unlock();

        return retName;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string cPropertyList::SizeString() const
    {
        return utilities::math::IntToStr( size() );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyList::PulseChanged()
    {
        // Emitujemy sygnaly zmiany wszystkich propertisow
        for( auto temp : m_vMainPropertyList )
        {
            if ( temp )
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
    void cPropertyList::CommitChanges()
    {
        m_Mutex.Lock();
        for( auto temp : m_vMainPropertyList )
        {
            if ( temp )
            {
                temp->CommitChanges();
            }
        }
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyList::Enable( bool val )
    {
        // Po wszystkich propertisach ustawiamy nowy stan
        m_Mutex.Lock();
        for( auto temp : m_vMainPropertyList )
        {
            if ( temp )
            {
                temp->Info().Enable( val );
            }
        }
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyList::RegisterProperty( PropertyBase* prop )
    {
        m_Mutex.Lock();
        m_vMainPropertyList.push_back( prop );
        prop->signalChanged.connect(this, &cPropertyList::slotPropertyChanged       );
        prop->signalChanged.connect(this, &cPropertyList::slotPropertyChangedGlobal );
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyList::UnRegisterProperty( PropertyBase* prop )
    {
        m_Mutex.Lock();
        // Po wszystkic1h zarejestrowanych parametrach
        for ( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
        {
            PropertyBase* temp = m_vMainPropertyList.at(n);
            if ( temp && temp->Name() == prop->Name() )
            {
                // Wywalamy z listy
                temp->signalChanged.disconnect( this );
                m_vMainPropertyList.erase(m_vMainPropertyList.begin() + n);
                break;
            }
        }
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyList::ClearPropertyList()
    {
        m_Mutex.Lock();
        m_vMainPropertyList.clear();
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool cPropertyList::IsPropertyUnique( const std::string& name ) const
    {
        int octcnt = 0;

        m_Mutex.Lock();
        for( auto temp : m_vMainPropertyList )
        {
            if ( temp && temp->Name() == name )
            {
                octcnt++;
            }
        }
        m_Mutex.Unlock();

        if (octcnt == 1 ) return true;
        else return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase* cPropertyList::GetObjectFieldValue( int cnt )
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
    int cPropertyList::GetObjectFieldCnt() const
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
    PropertyIterator cPropertyList::begin() throw()
    {
        return PropertyIterator( *this, 0 );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyIterator cPropertyList::end() throw()
    {
        return PropertyIterator( *this, GetObjectFieldCnt() );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int cPropertyList::size() const
    {
        return GetObjectFieldCnt();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyList::slotPropertyChangedGlobal( PropertyBase* prop )
    {
        signalPropertyChanged.Emit( prop );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyList::slotPropertyChanged( PropertyBase* prop )
    {
        #ifdef SERIALIZABLE_USE_WXWIDGETS
        wxUpdatePropertyValue( prop );
        #endif
    }
}
