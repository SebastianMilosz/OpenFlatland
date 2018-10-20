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
        public:
                     cSerializable( const std::string& name, cSerializableInterface* parent = NULL );
            virtual ~cSerializable();

            bool IsNameUnique( const std::string& name, bool checkParent = false ) const;

            cSerializablePath&       Path();
            cSerializableStorage&    Storage();
            cSerializableSelectable& Selection();
            cSerializableLua&        Script();
            cPropertyManager&        PropertyManager();
            cSerializableChildList&  ChildList();
            cSerializableIdentity&   Identity();

            std::string              SizeString() const;
            void                     PulseChanged( bool fullTree = false );
            void                     CommitChanges();
            void                     Enable( bool val );

        private:
            cSerializablePath       m_SerializablePath;
            cSerializableStorage    m_SerializableStorage;
            cSerializableSelectable m_SerializableSelectable;
            cSerializableLua        m_SerializableLua;
            cPropertyManager        m_PropertyManager;
            cSerializableChildList  m_childList;
            cSerializableIdentity   m_Identity;
    };
}

#endif
