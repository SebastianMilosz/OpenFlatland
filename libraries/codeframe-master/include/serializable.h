#ifndef _CSERIALIZABLE_H
#define _CSERIALIZABLE_H

#include <sigslot.h>

#include "serializablelua.h"
#include "instancemanager.h"
#include "serializablestorage.h"

namespace codeframe
{
    enum eShareLevel { ShareThis = 0, ShareFull = 1 };

    /*****************************************************************************/
    /**
      * @brief
      * @author Sebastian Milosz
      * @version 1.0
      * @note cSetializable
     **
    ******************************************************************************/
    class cSerializable : public cSerializableLua, public cInstanceManager, public cSerializableStorage
    {
        friend class Property;

        public:
                             cSerializable( std::string name, cSerializable* parent = NULL );
            virtual         ~cSerializable();

            void             SetName      ( std::string name );
            cSerializable*   ShareLevel   ( eShareLevel level = ShareFull );
            cSerializable*   LoadFromFile ( std::string filePath, std::string container = "", bool createIfNotExist = false );
            cSerializable*   LoadFromXML  ( cXML        xml,      std::string container = "" );
            cSerializable*   SaveToFile   ( std::string filePath, std::string container = "" );
            cXML             SaveToXML    (                       std::string container = "", int mode = 0 );

            bool             IsPropertyUnique( std::string name ) const;
            bool             IsNameUnique    ( std::string name, bool checkParent = false ) const;
            std::string      Path() const;
            std::string      ObjectName() const;
            std::string      SizeString() const;
            cSerializable*   Parent()     const;
            cSerializable*   GetRootObject      (                  );
            cSerializable*   GetObjectFromPath  ( std::string path );
            cSerializable*   GetChildByName     ( std::string name );
            void             RegisterProperty   ( Property*   prop );
            void             UnRegisterProperty ( Property*   prop );
            void             ClearPropertyList  (                  );
            Property*        GetPropertyByName  ( std::string name );
            Property*        GetPropertyById    ( uint32_t    id   );
            Property*        GetPropertyFromPath( std::string path );
            std::string      GetNameById        ( uint32_t    id   ) const;
            void             PulseChanged       ( bool fullTree = false );
            void             CommitChanges      (                  );
            void             Enable             ( int val = -1     );
            void             ParentUnbound      ();

        public: // Sygnaly
            signal1<Property*> signalPropertyChanged;                   ///< Emitowany gdy propertis został zmieniony razem z wskaznikiem na niego
            signal1<Property*> signalPropertyUpdateFail;                ///< Emitowany gdy oczekiwano ustawienia propertisa

        protected: // Sloty
            virtual void slotPropertyChangedGlobal( Property* prop );   ///<

            void    EnterPulseState()    { m_pulseState = true;  }
            void    LeavePulseState()    { m_pulseState = false; }
            bool    IsPulseState() const { return  m_pulseState; }

        private:
            virtual void slotPropertyChanged( Property* prop );   ///<

            int             m_delay;
            cSerializable*  m_parent;
            eShareLevel     m_shareLevel;
            std::string     m_sContainerName;
            bool            m_pulseState;
    };
}

#endif
