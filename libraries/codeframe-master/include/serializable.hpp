#ifndef _CSERIALIZABLE_H
#define _CSERIALIZABLE_H

#include "serializableinterface.hpp"
#include "serializablepropertybase.hpp"
#include "serializablestorage.hpp"
#include "instancemanager.hpp"

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
    class cSerializable :
        public cInstanceManager,
        public cSerializableInterface
    {
        friend class PropertyBase;

        public:
                             cSerializable( const std::string& name, cSerializableInterface* parent = NULL );
            virtual         ~cSerializable();

            void             SetName      ( const std::string& name );

            bool                     IsNameUnique    ( const std::string& name, bool checkParent = false ) const;

            cSerializablePath&       Path();
            cSerializableStorage&    Storage();
            cSerializableSelectable& Selection();
            cSerializableLua&        Script();
            cPropertyManager&        PropertyManager();

            std::string              ObjectName( bool idSuffix = true ) const;
            std::string              SizeString() const;
            void                     PulseChanged       ( bool fullTree = false );
            void                     CommitChanges      (                  );
            void                     Enable             ( bool val         );

        protected: // Sloty
            void    EnterPulseState()    { m_pulseState = true;  }
            void    LeavePulseState()    { m_pulseState = false; }
            bool    IsPulseState() const { return  m_pulseState; }

        private:
            cSerializablePath       m_SerializablePath;
            cSerializableStorage    m_SerializableStorage;
            cSerializableSelectable m_SerializableSelectable;
            cSerializableLua        m_SerializableLua;
            cPropertyManager        m_PropertyManager;

            std::string             m_sContainerName;
            bool                    m_pulseState;
    };
}

#endif
