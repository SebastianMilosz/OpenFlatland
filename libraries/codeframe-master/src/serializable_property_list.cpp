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
        auto it = m_PropertyMap.find(name);
        if (it != m_PropertyMap.end())
        {
            return smart_ptr<PropertyNode>( new PropertySelection(it->second) );
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
        for( auto element : m_PropertyMap )
        {
            if( element.second && element.second->Id() == id )
            {
                return smart_ptr<PropertyNode>( new PropertySelection( element.second ) );
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
    smart_ptr<PropertyNode> cPropertyList::GetPropertyFromPath(const std::string& path)
    {
        std::string::size_type found( path.find_last_of(".") );
        std::string objPath( path.substr( 0, found ) );
        std::string propertyName( path.substr( found+1 ) );

        smart_ptr<ObjectSelection> objectSelection = m_sint.Path().GetObjectFromPath( objPath );

        if ( smart_ptr_isValid( objectSelection ) )
        {
            if ( objectSelection->GetNodeCount() > 1U )
            {
                smart_ptr<PropertyMultipleSelection> propMultiNode( new PropertyMultipleSelection() );

                for ( auto obj : *objectSelection )
                {
                    smart_ptr<PropertyNode> node = obj->PropertyList().GetPropertyByName( propertyName );
                    if ( smart_ptr_isValid(node) )
                    {
                        propMultiNode->Add( node );
                    }
                }
                return propMultiNode;
            }
            else
            {
                smart_ptr<PropertyNode> propNode = objectSelection->Property( propertyName );
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

        for( auto element : m_PropertyMap )
        {
            if ( element.second && element.second->Id() == id )
            {
                retName = element.second->Name();
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
        for( auto element : m_PropertyMap )
        {
            if ( element.second )
            {
                element.second->PulseChanged();
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
        for( auto element : m_PropertyMap )
        {
            if ( element.second )
            {
                element.second->CommitChanges();
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
        for( auto element : m_PropertyMap )
        {
            if ( element.second )
            {
                element.second->Info().Enable( val );
            }
        }
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyList::RegisterProperty( const std::string& name, PropertyBase* prop )
    {
        m_Mutex.Lock();
        m_PropertyMap.emplace( name, prop );
        prop->signalChanged.connect(this, &cPropertyList::slotPropertyChanged       );
        prop->signalChanged.connect(this, &cPropertyList::slotPropertyChangedGlobal );
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyList::UnRegisterProperty( const std::string& name, PropertyBase* prop )
    {
        m_Mutex.Lock();
        auto it = m_PropertyMap.find(name);
        if (it != m_PropertyMap.end())
        {
            it->second->signalChanged.disconnect( this );
            m_PropertyMap.erase(it);
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
        m_PropertyMap.clear();
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool cPropertyList::IsPropertyUnique( const std::string& name ) const
    {
        bool retVal = true;

        m_Mutex.Lock();
        auto it = m_PropertyMap.find(name);
        if (it != m_PropertyMap.end())
        {
            retVal = false;
        }
        m_Mutex.Unlock();

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyIterator cPropertyList::begin() throw()
    {
        return PropertyIterator( m_PropertyMap.begin() );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyIterator cPropertyList::end() throw()
    {
        return PropertyIterator( m_PropertyMap.end() );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int cPropertyList::size() const
    {
        m_Mutex.Lock();
        int retSize = m_PropertyMap.size();
        m_Mutex.Unlock();

        return retSize;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyList::slotPropertyChangedGlobal( PropertyNode* prop )
    {
        signalPropertyChanged.Emit( prop );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPropertyList::slotPropertyChanged( PropertyNode* prop )
    {
        #ifdef SERIALIZABLE_USE_WXWIDGETS
        wxUpdatePropertyValue( prop );
        #endif
    }
}
