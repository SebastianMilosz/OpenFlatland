#ifndef SERIALIZABLEPROPERTYMANAGER_HPP_INCLUDED
#define SERIALIZABLEPROPERTYMANAGER_HPP_INCLUDED

#include <sigslot.h>
#include <string>
#include <vector>

#include "serializablepropertyiterator.hpp"
#include "serializablepropertybase.hpp"

namespace codeframe
{
    class cSerializableInterface;
    class PropertyBase;

    class cPropertyManager : public sigslot::has_slots<>
    {
        friend class PropertyIterator;

        public:
             cPropertyManager( cSerializableInterface& sint );
            ~cPropertyManager();

            PropertyBase* GetPropertyByName  ( const std::string& name );
            PropertyBase* GetPropertyById    ( uint32_t    id   );
            PropertyBase* GetPropertyFromPath( const std::string& path );
            std::string   GetNameById        ( uint32_t    id   ) const;

            void PulseChanged();
            void CommitChanges();
            void Enable( bool val );
            void RegisterProperty  ( PropertyBase* prop );
            void UnRegisterProperty( PropertyBase* prop );
            void ClearPropertyList ();
            bool IsPropertyUnique( const std::string& name ) const;

            /// Zwraca wartosc pola do serializacji
            PropertyBase* GetObjectFieldValue( int cnt );

            /// Zwraca ilosc skladowych do serializacji
            int GetObjectFieldCnt() const;

            /// Iterators
            PropertyIterator begin() throw();
            PropertyIterator end()   throw();
            int              size()  const;

            signal1<PropertyBase*> signalPropertyChanged;                   ///< Emitowany gdy propertis zosta³ zmieniony razem z wskaznikiem na niego
            signal1<PropertyBase*> signalPropertyUpdateFail;                ///< Emitowany gdy oczekiwano ustawienia propertisa

        private:
            void slotPropertyChangedGlobal( PropertyBase* prop );
            void slotPropertyChanged( PropertyBase* prop );

            cSerializableInterface& m_sint;

            /// Kontenet zawierajacy wskazniki do parametrow
            std::vector<PropertyBase*>  m_vMainPropertyList;

            ///
            PropertyBase m_dummyProperty;

            ///
            WrMutex m_Mutex;
    };
}

#endif // SERIALIZABLEPROPERTYMANAGER_HPP_INCLUDED
