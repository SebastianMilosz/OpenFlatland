#include "serializablecontainer.hpp"

#include <LoggerUtilities.h>

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSerializableContainer::cSerializableContainer( const std::string& name, ObjectNode* parentObject ) :
    cSerializable( name, parentObject ),
    m_selected( smart_ptr<cSerializable>(NULL) ),
    m_size( 0 )
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSerializableContainer::~cSerializableContainer()
{
    Dispose();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int cSerializableContainer::Count() const
{
    return m_size;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableContainer::CreateRange( const std::string& className, const std::string& objName, int range )
{
    for(int i = 0; i < range; i++)
    {
        if( smart_ptr_isValid( Create( className, objName ) ) == false )
        {
            throw std::runtime_error( "cSerializableContainer::Create return NULL" );
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool cSerializableContainer::IsName( const std::string& name )
{
    for(typename std::vector< smart_ptr<cSerializable> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); ++it)
    {
        smart_ptr<cSerializable> sptr = *it;

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
std::string cSerializableContainer::CreateUniqueName( const std::string& nameBase )
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
bool cSerializableContainer::Dispose( unsigned int id )
{
    if ( m_containerVector.size() <= id ) return false;

    smart_ptr<cSerializable> obj = m_containerVector[ id ];

    if ( smart_ptr_isValid( obj ) == true )
    {
        m_containerVector[ id ]->Selection().DisconectFromContainer();
        m_containerVector[ id ] = smart_ptr<cSerializable>(NULL);

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
bool cSerializableContainer::Dispose( const std::string& objName )
{
    // Unimplemented
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool cSerializableContainer::DisposeByBuildType( eBuildType serType, cIgnoreList ignore )
{
    for ( typename std::vector< smart_ptr<cSerializable> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); )
    {
        smart_ptr<cSerializable> sptr = *it;

        if ( smart_ptr_isValid( sptr ) && sptr->BuildType() == serType && ignore.IsIgnored( smart_ptr_getRaw( sptr ) ) == false )
        {
            sptr->Selection().DisconectFromContainer();
            *it = smart_ptr<cSerializable>(NULL);

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
bool cSerializableContainer::Dispose( smart_ptr<ObjectNode> obj )
{
    for(typename std::vector< smart_ptr<cSerializable> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); ++it)
    {
        smart_ptr<cSerializable> sptr = *it;

        if( smart_ptr_isValid( sptr ) && smart_ptr_isValid( obj ) )
        {
            if( sptr->Identity().ObjectName() == obj->Identity().ObjectName() )
            {
                sptr->Selection().DisconectFromContainer();
                *it = smart_ptr<cSerializable>();
                if( m_size ) m_size--;
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
bool cSerializableContainer::Dispose()
{
    if( m_containerVector.size() == 0 ) return true;    // Pusty kontener zwracamy prawde bo nie ma nic do usuwania

    for(typename std::vector< smart_ptr<cSerializable> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); ++it)
    {
        smart_ptr<cSerializable> obj = *it;

        // Usuwamy tylko jesli nikt inny nie korzysta z obiektu
        if( smart_ptr_getCount( obj ) <= 2 )
        {
            obj->Selection().DisconectFromContainer();
            obj = smart_ptr<cSerializable>(NULL);
        }
        else // Nie mozna usunac obiektu
        {
            return false;
        }
    }

    m_containerVector.clear();
    m_size = 0;

    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool cSerializableContainer::IsInRange( unsigned int cnt ) const
{
    return (bool)( cnt < m_containerVector.size() );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectNode> cSerializableContainer::operator[]( int i )
{
    return Get( i );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool cSerializableContainer::Select( int pos )
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
bool cSerializableContainer::IsSelected()
{
    return smart_ptr_isValid( m_selected );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectNode> cSerializableContainer::GetSelected()
{
    return m_selected;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectNode> cSerializableContainer::Get( int id )
{
    if( IsInRange( id ) )
    {
        smart_ptr<ObjectNode> obj = m_containerVector.at( id );

        if( smart_ptr_isValid( obj ) == true )
        {
            return obj;
        }
    }

    return smart_ptr<ObjectNode>( NULL );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int cSerializableContainer::Add( smart_ptr<cSerializable> classType, int pos )
{
    return InsertObject( classType, pos );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int cSerializableContainer::InsertObject( smart_ptr<cSerializable> classType, int pos )
{
    // pos == -1 oznacza pierwszy lepszy
    bool found  = false;
    int  retPos = -1;

    if( pos < static_cast<int>(m_containerVector.size()) )
    {
        // Szukamy bezposrednio
        if( pos >= 0 )
        {
            smart_ptr<cSerializable> tmp = m_containerVector[ pos ];

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
            for(typename std::vector< smart_ptr<cSerializable> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); ++it)
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
    if( NULL == classType->Path().Parent() )
    {
        ObjectNode* serPar = static_cast<ObjectNode*>( this );
        classType->Path().ParentBound( serPar );
    }

    classType->Selection().ConectToContainer<cSerializableContainer>( this, classType );
    classType->Identity().SetId( retPos );

    m_size++;

    return retPos;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableContainer::slotSelectionChanged( smart_ptr<ObjectNode> obj )
{
    cSerializable* serializableObjectNew = static_cast<cSerializable*>( smart_ptr_getRaw(obj) );

    if ( (cSerializable*)NULL != serializableObjectNew )
    {
        std::string name = serializableObjectNew->Identity().ObjectName();

        if ( serializableObjectNew->Selection().IsSelected() == true )
        {
            cSerializable* serializableObjectSel = static_cast<cSerializable*>( smart_ptr_getRaw(m_selected) );

            if( serializableObjectSel != serializableObjectNew )
            {
                if ( (cSerializable*)NULL != serializableObjectSel )
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
