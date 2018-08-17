#ifndef _CSERIALIZABLE_H
#define _CSERIALIZABLE_H

#include <sigslot.h>

#include "serializablelua.hpp"
#include "serializablestorage.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
      * @author Sebastian Milosz
      * @version 1.0
      * @note cSetializable
     **
    ******************************************************************************/
    class cSerializable : public cSerializableLua, public cSerializableStorage
    {
        friend class PropertyBase;

        public:
                             cSerializable( std::string const& name, cSerializableInterface* parent = NULL );
            virtual         ~cSerializable();

            void             SetName      ( std::string const& name );
            cSerializable*   ShareLevel   ( eShareLevel level = ShareFull );
            cSerializable*   LoadFromFile ( std::string const& filePath, std::string const& container = "", bool createIfNotExist = false );
            cSerializable*   LoadFromXML  ( cXML xml, std::string const& container = "" );
            cSerializable*   SaveToFile   ( std::string const& filePath, std::string const& container = "" );
            cXML             SaveToXML    ( std::string const& container = "", int mode = 0 );

            bool                    IsPropertyUnique( std::string const& name ) const;
            bool                    IsNameUnique    ( std::string const& name, bool checkParent = false ) const;
            std::string             Path() const;
            std::string             ObjectName( bool idSuffix = true ) const;
            std::string             SizeString() const;
            cSerializableInterface* Parent()     const;
            cSerializableInterface* GetRootObject      (                  );
            cSerializableInterface* GetObjectFromPath  ( std::string const& path );
            cSerializableInterface* GetChildByName     ( std::string const& name );
            void                    RegisterProperty   ( PropertyBase*   prop );
            void                    UnRegisterProperty ( PropertyBase*   prop );
            void                    ClearPropertyList  (                  );
            PropertyBase*           GetPropertyByName  ( std::string const& name );
            PropertyBase*           GetPropertyById    ( uint32_t    id   );
            PropertyBase*           GetPropertyFromPath( std::string const& path );
            std::string             GetNameById        ( uint32_t    id   ) const;
            void                    PulseChanged       ( bool fullTree = false );
            void                    CommitChanges      (                  );
            void                    Enable             ( bool val         );
            void                    ParentUnbound      ();
            void                    ParentBound        ( cSerializableInterface* obj );

        public: // @todo move to Property Sygnaly
            signal1<PropertyBase*> signalPropertyChanged;                   ///< Emitowany gdy propertis został zmieniony razem z wskaznikiem na niego
            signal1<PropertyBase*> signalPropertyUpdateFail;                ///< Emitowany gdy oczekiwano ustawienia propertisa

        protected: // Sloty
            virtual void slotPropertyChangedGlobal( PropertyBase* prop );   ///<

            void    EnterPulseState()    { m_pulseState = true;  }
            void    LeavePulseState()    { m_pulseState = false; }
            bool    IsPulseState() const { return  m_pulseState; }

        private:
            virtual void slotPropertyChanged( PropertyBase* prop );   ///<

            int                     m_delay;
            cSerializableInterface* m_parent;
            std::string             m_sContainerName;
            bool                    m_pulseState;
    };
}

#endif
