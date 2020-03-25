#ifndef PROPERTY_LIST_HPP_INCLUDED
#define PROPERTY_LIST_HPP_INCLUDED

#include <ThreadUtilities.h>
#include <sigslot.h>
#include <string>
#include <vector>
#include <smartpointer.h>

#include "serializable_property_iterator.hpp"
#include "serializable_property_base.hpp"

namespace codeframe
{
    class ObjectNode;
    class PropertyNode;

    class cPropertyList : public sigslot::has_slots<>
    {
        friend class PropertyIterator;

        public:
             cPropertyList( ObjectNode& sint );
            ~cPropertyList();

            smart_ptr<PropertyNode> GetPropertyByName  ( const std::string& name );
            smart_ptr<PropertyNode> GetPropertyById    ( const uint32_t id   );
            smart_ptr<PropertyNode> GetPropertyFromPath( const std::string& path );

            std::string   GetNameById( uint32_t id ) const;
            std::string   SizeString() const;

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

            signal1<PropertyNode*> signalPropertyChanged;                   ///< Emitowany gdy propertis zosta³ zmieniony razem z wskaznikiem na niego
            signal1<PropertyNode*> signalPropertyUpdateFail;                ///< Emitowany gdy oczekiwano ustawienia propertisa

        private:
            void slotPropertyChangedGlobal( PropertyNode* prop );
            void slotPropertyChanged( PropertyNode* prop );

            ObjectNode& m_sint;

            /// Kontenet zawierajacy wskazniki do parametrow
            std::vector<PropertyBase*>  m_vMainPropertyList;

            ///
            PropertyBase m_dummyProperty;

            ///
            WrMutex m_Mutex;
    };
}

#endif // PROPERTY_LIST_HPP_INCLUDED
