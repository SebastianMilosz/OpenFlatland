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

        if ( fullTree )
        {
            m_childList.PulseChanged( fullTree );
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
        m_SerializablePath.ParentUnbound();
        m_PropertyManager.ClearPropertyList();
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
      * @brief Zatwierdzenie wszystkich zmian obiektu i jego potomnych
     **
    ******************************************************************************/
    void cSerializable::CommitChanges()
    {
        m_PropertyManager.CommitChanges();
        m_childList.CommitChanges();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializable::Enable( bool val )
    {
        m_PropertyManager.Enable( val );
        m_childList.Enable( val );
    }

}
