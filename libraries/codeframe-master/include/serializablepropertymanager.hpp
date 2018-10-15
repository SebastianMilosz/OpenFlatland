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
        private:
            ///< Kontenet zawierajacy wskazniki do parametrow
            std::vector<PropertyBase*>  m_vMainPropertyList;
    };
}

#endif // SERIALIZABLEPROPERTYMANAGER_HPP_INCLUDED
