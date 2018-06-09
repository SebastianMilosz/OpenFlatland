#ifndef _CSERIALIZABLE_H
#define _CSERIALIZABLE_H

#include <sigslot.h>

#include "serializablelua.h"
#include "serializablestorage.h"

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
        friend class Property;

        public:
                             cSerializable( std::string const& name, cSerializable* parent = NULL );
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
            std::string             ObjectName() const;
            std::string             SizeString() const;
            cSerializableInterface* Parent()     const;
            cSerializableInterface* GetRootObject      (                  );
            cSerializableInterface* GetObjectFromPath  ( std::string const& path );
            cSerializableInterface* GetChildByName     ( std::string const& name );
            void                    RegisterProperty   ( Property*   prop );
            void                    UnRegisterProperty ( Property*   prop );
            void                    ClearPropertyList  (                  );
            Property*               GetPropertyByName  ( std::string const& name );
            Property*               GetPropertyById    ( uint32_t    id   );
            Property*               GetPropertyFromPath( std::string const& path );
            std::string             GetNameById        ( uint32_t    id   ) const;
            void                    PulseChanged       ( bool fullTree = false );
            void                    CommitChanges      (                  );
            void                    Enable             ( bool val         );
            void                    ParentUnbound      ();
            void                    ParentBound        ( cSerializableInterface* obj );

        public: // @todo move to Property Sygnaly
            signal1<Property*> signalPropertyChanged;                   ///< Emitowany gdy propertis został zmieniony razem z wskaznikiem na niego
            signal1<Property*> signalPropertyUpdateFail;                ///< Emitowany gdy oczekiwano ustawienia propertisa

        protected: // Sloty
            virtual void slotPropertyChangedGlobal( Property* prop );   ///<

            void    EnterPulseState()    { m_pulseState = true;  }
            void    LeavePulseState()    { m_pulseState = false; }
            bool    IsPulseState() const { return  m_pulseState; }

        private:
            virtual void slotPropertyChanged( Property* prop );   ///<

            int                     m_delay;
            cSerializableInterface* m_parent;
            std::string             m_sContainerName;
            bool                    m_pulseState;
    };
}

#endif
