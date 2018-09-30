#ifndef _CSERIALIZABLE_H
#define _CSERIALIZABLE_H

#include <sigslot.h>

#include "serializablepropertybase.hpp"
#include "serializablelua.hpp"
#include "serializablestorage.hpp"
#include "serializableselectable.hpp"

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
    class cSerializable : public cSerializableLua, public cSerializableStorage, public cSerializableSelectable
    {
        friend class PropertyBase;

        public:
                             cSerializable( const std::string& name, cSerializableInterface* parent = NULL );
            virtual         ~cSerializable();

            void             SetName      ( const std::string& name );
            cSerializable*   ShareLevel   ( eShareLevel level = ShareFull );
            cSerializable*   LoadFromFile ( const std::string& filePath, const std::string& container = "", bool createIfNotExist = false );
            cSerializable*   LoadFromXML  ( cXML xml, const std::string& container = "" );
            cSerializable*   SaveToFile   ( const std::string& filePath, const std::string& container = "" );
            cXML             SaveToXML    ( const std::string& container = "", int mode = 0 );

            bool                    IsPropertyUnique( const std::string& name ) const;
            bool                    IsNameUnique    ( const std::string& name, bool checkParent = false ) const;
            std::string             Path() const;
            std::string             ObjectName( bool idSuffix = true ) const;
            std::string             SizeString() const;
            cSerializableInterface* Parent()     const;
            cSerializableInterface* GetRootObject      (                  );
            cSerializableInterface* GetObjectFromPath  ( const std::string& path );
            cSerializableInterface* GetChildByName     ( const std::string& name );
            void                    RegisterProperty   ( PropertyBase*   prop );
            void                    UnRegisterProperty ( PropertyBase*   prop );
            void                    ClearPropertyList  (                  );
            PropertyBase*           GetPropertyByName  ( const std::string& name );
            PropertyBase*           GetPropertyById    ( uint32_t    id   );
            PropertyBase*           GetPropertyFromPath( const std::string& path );
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
