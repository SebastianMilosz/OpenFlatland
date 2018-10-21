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
    cSerializable::cSerializable( const std::string& name, cSerializableInterface* parent ) :
        m_SerializablePath( *this ),
        m_SerializableStorage( *this ),
        m_SerializableSelectable( *this ),
        m_SerializableLua( *this ),
        m_PropertyManager( *this ),
        m_Identity( name )
    {
        m_SerializablePath.ParentBound( parent );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializable::PulseChanged( bool fullTree )
    {
        m_Identity.EnterPulseState();

        m_PropertyManager.PulseChanged();

        m_Identity.LeavePulseState();

        if( fullTree )
        {
            // Zmuszamy dzieci do aktualizacji
            for( cSerializableChildList::iterator it = this->ChildList().begin(); it != this->ChildList().end(); ++it )
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
    cSerializable::~cSerializable()
    {
        // Wyrejestrowywujemy sie u rodzica
        Path().ParentUnbound();
        PropertyManager().ClearPropertyList();
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
    cSerializableSelectable& cSerializable::Selection()
    {
        return m_SerializableSelectable;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableLua& cSerializable::Script()
    {
        return m_SerializableLua;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cPropertyManager& cSerializable::PropertyManager()
    {
        return m_PropertyManager;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableChildList& cSerializable::ChildList()
    {
        return m_childList;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableIdentity& cSerializable::Identity()
    {
        return m_Identity;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string cSerializable::SizeString() const
    {
        return utilities::math::IntToStr( m_PropertyManager.size() );
    }

    /*****************************************************************************/
    /**
      * @brief Zatwierdzenie wszystkich zmian obiektu i jego potomnych
     **
    ******************************************************************************/
    void cSerializable::CommitChanges()
    {
        PropertyManager().CommitChanges();

        //Mutex().Lock();
        for( cSerializableChildList::iterator it = ChildList().begin(); it != ChildList().end(); ++it )
        {
            cSerializableInterface* iser = *it;
            if( iser )
            {
                iser->CommitChanges();
            }
        }
        //Mutex().Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializable::Enable( bool val )
    {
        PropertyManager().Enable( val );

        //Mutex().Lock();
        for( cSerializableChildList::iterator it = ChildList().begin(); it != ChildList().end(); ++it )
        {
            cSerializableInterface* iser = *it;
            if( iser )
            {
                iser->Enable( val );
            }
        }
        //Mutex().Unlock();
    }

}
