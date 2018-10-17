#ifndef SERIALIZABLEPROPERTYMANAGER_HPP_INCLUDED
#define SERIALIZABLEPROPERTYMANAGER_HPP_INCLUDED

#include <string>
#include <vector>

namespace codeframe
{
    class cSerializableInterface;
    class PropertyBase;

    class cPropertyManager
    {
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

            /// Zwraca wartosc pola do serializacji
            PropertyBase* GetObjectFieldValue( int cnt );

            /// Zwraca ilosc skladowych do serializacji
            int GetObjectFieldCnt() const;

            /// Iterators
            PropertyIterator begin() throw();
            PropertyIterator end()   throw();
            int              size()  const;

        private:
            /// Kontenet zawierajacy wskazniki do parametrow
            std::vector<PropertyBase*>  m_vMainPropertyList;
    };
}

#endif // SERIALIZABLEPROPERTYMANAGER_HPP_INCLUDED
