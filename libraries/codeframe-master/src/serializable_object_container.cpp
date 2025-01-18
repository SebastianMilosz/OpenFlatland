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
    m_selected( smart_ptr<Object>( nullptr ) )
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ObjectContainer::ObjectContainer( const std::string& name, smart_ptr<ObjectNode> parentObject ) :
    Object( name, parentObject ),
    m_selected( smart_ptr<Object>( nullptr ) )
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
    return m_containerVector.size();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ObjectContainer::CreateRange( const std::string& className, const std::string& objName, const int range )
{
    for ( int i = 0; i < range; i++ )
    {
        if ( smart_ptr_isValid( Create( className, objName ) ) == false )
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
    for ( auto& sptr : m_containerVector )
    {
        if ( smart_ptr_isValid( sptr ) == true )
        {
            std::string inContainerName( sptr->Identity().ObjectName() );

            if ( name == inContainerName )
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
    std::string uniqueName( nameBase );

    for ( unsigned int curIter = 0U; curIter < MAXID; curIter++ )
    {
        std::string name( nameBase + utilities::math::IntToStr( curIter ) );

        if ( IsName( name ) == false )
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
bool ObjectContainer::Dispose( const unsigned int id )
{
    if ( m_containerVector.size() <= id )
    {
        return false;
    }

    smart_ptr<Object> obj = m_containerVector[ id ];

    if ( smart_ptr_isValid( obj ) == true )
    {
        m_containerVector[ id ]->Selection().DisconectFromContainer();
        m_containerVector[ id ] = smart_ptr<Object>(nullptr);
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
bool ObjectContainer::DisposeByBuildType( const eBuildType buildType, const cIgnoreList ignoreList )
{
    for ( auto it = m_containerVector.begin(); it != m_containerVector.end(); )
    {
        smart_ptr<Object> sptr = *it;

        if (
             smart_ptr_isValid( sptr ) &&
             sptr->BuildType() == buildType &&
             ignoreList.IsIgnored( smart_ptr_getRaw( sptr ) ) == false
           )
        {
            sptr->Selection().DisconectFromContainer();
            sptr->Unbound();

            signalContainerSelectionChanged.Emit( sptr );

            it = m_containerVector.erase(it);
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
    for ( auto it = m_containerVector.begin(); it != m_containerVector.end(); ++it )
    {
        smart_ptr<Object> sptr = *it;

        if ( smart_ptr_isValid( sptr ) && smart_ptr_isValid( obj ) )
        {
            if ( sptr->Identity().ObjectName() == obj->Identity().ObjectName() )
            {
                sptr->Selection().DisconectFromContainer();
                sptr->Unbound();

                signalContainerSelectionChanged.Emit( sptr );

                it = m_containerVector.erase(it);

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
    if( m_containerVector.size() != 0U )
    {
        for ( auto it = m_containerVector.begin(); it != m_containerVector.end(); ++it )
        {
            smart_ptr<Object> obj = *it;

            if ( smart_ptr_getCount( obj ) <= 2 )
            {
                obj->Selection().DisconectFromContainer();
                obj = smart_ptr<Object>( nullptr );
            }
            else
            {
                return false;
            }
        }

        m_containerVector.clear();
    }

    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool ObjectContainer::IsInRange( const unsigned int cnt ) const
{
    return (bool)( cnt < m_containerVector.size() );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectContainer::operator[]( const unsigned int i )
{
    return Child(i);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectContainer::operator[]( const std::string& name )
{
    return Child(name);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectContainer::Child( const unsigned int i )
{
    return smart_ptr<ObjectSelection>( new ObjectSelection( Get( i ) ) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ObjectSelection> ObjectContainer::Child( const std::string& name )
{
    for (const auto& iteam: m_containerVector)
    {
        if (smart_ptr_isValid( iteam ) && iteam->Identity().ObjectName() == name)
        {
            return smart_ptr<ObjectSelection>( new ObjectSelection( iteam ) );
        }
    }

    return smart_ptr<ObjectSelection>( nullptr );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool ObjectContainer::Select( const int pos )
{
    if ( IsInRange( pos ) )
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
smart_ptr<ObjectNode> ObjectContainer::Get( const int id )
{
    if ( IsInRange( id ) )
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
int ObjectContainer::Add( smart_ptr<Object> classType, const int pos )
{
    return InsertObject( classType, pos );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int ObjectContainer::InsertObject( smart_ptr<Object> classType, const int pos )
{
    // pos == -1 oznacza pierwszy lepszy
    bool found  = false;
    int  retPos = -1;

    if ( pos < static_cast<int>(m_containerVector.size()) )
    {
        // Szukamy bezposrednio
        if ( pos >= 0 )
        {
            smart_ptr<Object> tmp = m_containerVector[ pos ];

            if ( smart_ptr_isValid( tmp ) )
            {
                m_containerVector[ pos ] = classType;
                found = true;
            }
        }

        // Szukamy wolnego miejsca
        if ( found == false )
        {
            // Po calym wektorze szukamy pustych miejsc
            for ( auto it = m_containerVector.begin(); it != m_containerVector.end(); ++it )
            {
                smart_ptr<ObjectNode> obj = *it;

                if ( smart_ptr_isValid( obj ) == false )
                {
                    *it = classType;
                    found = true;
                    retPos = std::distance( m_containerVector.begin(), it );
                    break;
                }
            }
        }
    }

    // if out of range we add it at the end
    if ( found == false )
    {
        m_containerVector.push_back( classType );
        retPos = m_containerVector.size() - 1;
    }

    // If there is no parent we become one
    if ( nullptr == classType->Path().Parent() )
    {
        ObjectNode* serPar = static_cast<ObjectNode*>( this );
        classType->Path().ParentBound( smart_ptr_wild<ObjectNode>(serPar, [](ObjectNode* p) {}) );
    }

    classType->Selection().ConectToContainer<ObjectContainer>( this, classType );
    classType->Identity().SetId( retPos );

    ReferenceManager::ResolveReferences(*(ObjectNode*)this);

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
        if ( serializableObjectNew->Selection().IsSelected() == true )
        {
            Object* serializableObjectSel = static_cast<Object*>( smart_ptr_getRaw(m_selected) );

            if ( serializableObjectSel != serializableObjectNew )
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
            LOGGER( LOG_INFO << "Object Deselected: " << serializableObjectNew->Identity().ObjectName() );
        }
    }
}
