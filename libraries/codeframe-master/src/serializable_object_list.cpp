#include "serializable_object_list.hpp"
#include "serializable_object.hpp"

#include <cstdio> // std::remove

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cObjectList::cObjectList()
    {
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cObjectList::Register( smart_ptr<ObjectNode> child )
    {
        if ( smart_ptr_isValid(child) )
        {
            m_Mutex.Lock();
            m_childVector.push_back( child );
            m_Mutex.Unlock();
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cObjectList::UnRegister( smart_ptr<ObjectNode> child )
    {
        if ( smart_ptr_isValid(child) )
        {
            m_Mutex.Lock();
            m_childVector.erase(std::remove(m_childVector.begin(), m_childVector.end(), child), m_childVector.end());
            m_Mutex.Unlock();
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cObjectList::PulseChanged( bool fullTree )
    {
        m_Mutex.Lock();
        for ( auto it = begin(); it != end(); ++it )
        {
            smart_ptr<ObjectNode> iser = *it;

            if ( smart_ptr_isValid(iser) )
            {
                iser->PulseChanged( fullTree );
            }
        }
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cObjectList::CommitChanges()
    {
        m_Mutex.Lock();
        for ( auto it = begin(); it != end(); ++it )
        {
            smart_ptr<ObjectNode> iser = *it;

            if ( smart_ptr_isValid(iser) )
            {
                iser->CommitChanges();
            }
        }
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cObjectList::Enable( bool val )
    {
        m_Mutex.Lock();
        for ( auto it = begin(); it != end(); ++it )
        {
            smart_ptr<ObjectNode> iser = *it;

            if ( smart_ptr_isValid(iser) )
            {
                iser->Enable( val );
            }
        }
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<ObjectSelection> cObjectList::GetObjectByName( const std::string& name )
    {
        // Separate * symbol
        std::size_t foundRangeOpen  = name.find_last_of("[");

        // Multi selection
        if ( foundRangeOpen != std::string::npos )
        {
            if ( name.at( foundRangeOpen + 1U ) == '*' )
            {
                auto  nameCore( name.substr( 0, foundRangeOpen + 1U ) );

                smart_ptr<ObjectMultipleSelection> multipleSelection = smart_ptr<ObjectMultipleSelection>( new ObjectMultipleSelection );

                for ( auto it = begin(); it != end(); ++it )
                {
                    smart_ptr<ObjectNode> iser = *it;
                    auto  objectName( iser->Identity().ObjectName( true ) );
                    auto  refName( nameCore + ( std::to_string( (int)it ).append( "]" ) ) );

                    if ( objectName == refName )
                    {
                        multipleSelection->Add( iser );
                    }
                }

                return multipleSelection;
            }
        }

        // Single selection
        for ( auto it = begin(); it != end(); ++it )
        {
            smart_ptr<ObjectNode> iser = *it;
            auto objectName( iser->Identity().ObjectName( true ) );

            if ( objectName == name )
            {
                return smart_ptr<ObjectSelection>( new ObjectSelection( iser ) );
            }
        }
        return smart_ptr<ObjectSelection>(  nullptr );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<ObjectSelection> cObjectList::GetObjectById( const uint32_t id )
    {
        return smart_ptr<ObjectSelection>(nullptr);
    }
}
