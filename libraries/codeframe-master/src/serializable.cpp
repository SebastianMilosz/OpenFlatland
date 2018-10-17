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

        PropertyManager().PulseChanged();

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
    cSerializablePath& cSerializable::Path() const
    {
        return m_SerializablePath;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableStorage& cSerializable::Storage() const
    {
        return m_SerializableStorage;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableSelectable& cSerializable::Selection() const
    {
        return m_SerializableSelectable;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableLua& cSerializable::Script() const
    {
        return m_SerializableLua;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cPropertyManager& cSerializable::PropertyManager() const
    {
        return m_PropertyManager;
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
        return utilities::math::IntToStr( PropertyManager().size() );
    }

    /*****************************************************************************/
    /**
      * @brief Zatwierdzenie wszystkich zmian obiektu i jego potomnych
     **
    ******************************************************************************/
    void cSerializable::CommitChanges()
    {
        PropertyManager().CommitChanges();

        Lock();
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
        PropertyManager().Enable( val );

        Lock();

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
