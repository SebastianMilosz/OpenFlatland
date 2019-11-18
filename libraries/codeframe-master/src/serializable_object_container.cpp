#include "serializable_object_container.hpp"

#include <LoggerUtilities.h>

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectContainer::ObjectContainer( const std::string& name, ObjectNode* parentObject ) :
    Object( name, parentObject ),
    m_selected( smart_ptr<Object>( nullptr ) ),
    m_size( 0U )
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectContainer::~ObjectContainer()
{
    Dispose();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
unsigned int ObjectContainer::Count() const
{
    return m_size;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ObjectContainer::CreateRange( const std::string& className, const std::string& objName, int range )
{
    for(int i = 0; i < range; i++)
    {
        if( smart_ptr_isValid( Create( className, objName ) ) == false )
        {
            throw std::runtime_error( "ObjectContainer::Create return NULL" );
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool ObjectContainer::IsName( const std::string& name )
{
    for(typename std::vector< smart_ptr<Object> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); ++it)
    {
        smart_ptr<Object> sptr = *it;

        if( smart_ptr_isValid( sptr ) == true )
        {
            std::string inContainerName = sptr->Identity().ObjectName();

            if( name == inContainerName )
            {
                return true;
            }
        }
    }

    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string ObjectContainer::CreateUniqueName( const std::string& nameBase )
{
    std::string uniqueName  = nameBase;

    for( int curIter = 0; curIter < MAXID; curIter++ )
    {
        std::string name = nameBase + utilities::math::IntToStr( curIter );

        if( IsName( name ) == false )
        {
            uniqueName = name;
            break;
        }
    }

    return uniqueName;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool ObjectContainer::Dispose( unsigned int id )
{
    if ( m_containerVector.size() <= id ) return false;

    smart_ptr<Object> obj = m_containerVector[ id ];

    if ( smart_ptr_isValid( obj ) == true )
    {
        m_containerVector[ id ]->Selection().DisconectFromContainer();
        m_containerVector[ id ] = smart_ptr<Object>(nullptr);

        if ( m_size )
        {
            m_size--;
        }
        return true;
    }

    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool ObjectContainer::Dispose( const std::string& objName )
{
    // Unimplemented
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool ObjectContainer::DisposeByBuildType( eBuildType serType, cIgnoreList ignore )
{
    for ( typename std::vector< smart_ptr<Object> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); )
    {
        smart_ptr<Object> sptr = *it;

        if ( smart_ptr_isValid( sptr ) && sptr->BuildType() == serType && ignore.IsIgnored( smart_ptr_getRaw( sptr ) ) == false )
        {
            sptr->Selection().DisconectFromContainer();
            *it = smart_ptr<Object>( nullptr );

            if ( m_size )
            {
                m_size--;
            }
            signalContainerSelectionChanged.Emit( sptr );
        }
        else
        {
            it++;
        }
    }

    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool ObjectContainer::Dispose( smart_ptr<ObjectNode> obj )
{
    for(typename std::vector< smart_ptr<Object> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); ++it)
    {
        smart_ptr<Object> sptr = *it;

        if( smart_ptr_isValid( sptr ) && smart_ptr_isValid( obj ) )
        {
            if( sptr->Identity().ObjectName() == obj->Identity().ObjectName() )
            {
                sptr->Selection().DisconectFromContainer();
                *it = smart_ptr<Object>();
                if ( m_size )
                {
                    m_size--;
                }
                signalContainerSelectionChanged.Emit( sptr );
                return true;
            }
        }
    }

    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool ObjectContainer::Dispose()
{
    if( m_containerVector.size() == 0 ) return true;    // Pusty kontener zwracamy prawde bo nie ma nic do usuwania

    for(typename std::vector< smart_ptr<Object> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); ++it)
    {
        smart_ptr<Object> obj = *it;

        // Usuwamy tylko jesli nikt inny nie korzysta z obiektu
        if( smart_ptr_getCount( obj ) <= 2 )
        {
            obj->Selection().DisconectFromContainer();
            obj = smart_ptr<Object>( nullptr );
        }
        else // Nie mozna usunac obiektu
        {
            return false;
        }
    }

    m_containerVector.clear();
    m_size = 0U;

    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool ObjectContainer::IsInRange( unsigned int cnt ) const
{
    return (bool)( cnt < m_containerVector.size() );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectNode> ObjectContainer::operator[]( int i )
{
    return Get( i );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool ObjectContainer::Select( int pos )
{
    if( IsInRange( pos ) )
    {
        m_selected = Get( pos );
        signalContainerSelectionChanged.Emit( m_selected );
        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool ObjectContainer::IsSelected()
{
    return smart_ptr_isValid( m_selected );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectNode> ObjectContainer::GetSelected()
{
    return m_selected;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectNode> ObjectContainer::Get( int id )
{
    if( IsInRange( id ) )
    {
        smart_ptr<ObjectNode> obj = m_containerVector.at( id );

        if( smart_ptr_isValid( obj ) == true )
        {
            return obj;
        }
    }

    return smart_ptr<ObjectNode>( nullptr );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int ObjectContainer::Add( smart_ptr<Object> classType, int pos )
{
    return InsertObject( classType, pos );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int ObjectContainer::InsertObject( smart_ptr<Object> classType, int pos )
{
    // pos == -1 oznacza pierwszy lepszy
    bool found  = false;
    int  retPos = -1;

    if( pos < static_cast<int>(m_containerVector.size()) )
    {
        // Szukamy bezposrednio
        if( pos >= 0 )
        {
            smart_ptr<Object> tmp = m_containerVector[ pos ];

            if( smart_ptr_isValid( tmp ) )
            {
                m_containerVector[ pos ] = classType;
                found = true;
            }
        }

        // Szukamy wolnego miejsca
        if( found == false )
        {
            // Po calym wektorze szukamy pustych miejsc
            for(typename std::vector< smart_ptr<Object> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); ++it)
            {
                smart_ptr<ObjectNode> obj = *it;

                if( smart_ptr_isValid( obj ) == false )
                {
                    // Znalezlismy wiec zapisujemy
                    *it = classType;
                    found = true;
                    retPos = std::distance( m_containerVector.begin(), it );
                    break;
                }
            }
        }
    }

    // poza zakresem dodajemy do wektora nowa pozycje
    if( found == false )
    {
        m_containerVector.push_back( classType );
        retPos = m_containerVector.size() - 1;
    }

    // If there is no parent we become one
    if( nullptr == classType->Path().Parent() )
    {
        ObjectNode* serPar = static_cast<ObjectNode*>( this );
        classType->Path().ParentBound( serPar );
    }

    classType->Selection().ConectToContainer<ObjectContainer>( this, classType );
    classType->Identity().SetId( retPos );

    m_size++;

    return retPos;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ObjectContainer::slotSelectionChanged( smart_ptr<ObjectNode> obj )
{
    Object* serializableObjectNew = static_cast<Object*>( smart_ptr_getRaw(obj) );

    if ( (Object*)nullptr != serializableObjectNew )
    {
        std::string name = serializableObjectNew->Identity().ObjectName();

        if ( serializableObjectNew->Selection().IsSelected() == true )
        {
            Object* serializableObjectSel = static_cast<Object*>( smart_ptr_getRaw(m_selected) );

            if( serializableObjectSel != serializableObjectNew )
            {
                if ( (Object*)nullptr != serializableObjectSel )
                {
                    serializableObjectSel->Selection().Select( false );
                }
            }

            m_selected = obj;

            signalContainerSelectionChanged.Emit( m_selected );
        }
        else
        {
            LOGGER( LOG_INFO << "Object Deselected: " << name );
        }
    }
}
