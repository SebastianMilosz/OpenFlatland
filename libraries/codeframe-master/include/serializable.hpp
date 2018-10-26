#ifndef _CSERIALIZABLE_H
#define _CSERIALIZABLE_H

#include "serializableinterface.hpp"
#include "serializablepropertybase.hpp"
#include "serializablestorage.hpp"

#define CODEFRAME_META_CLASS_NAME(p) public: std::string Class() const { return p; }
#define CODEFRAME_META_CONSTRUCT_PATERN(p) public: std::string ConstructPatern() const { return p; }
#define CODEFRAME_META_BUILD_ROLE(p) public: codeframe::eBuildRole Role() const { return p; }
#define CODEFRAME_META_BUILD_TYPE(p) public: codeframe::eBuildType BuildType() const { return p; }

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
    class cSerializable : public cSerializableInterface
    {
        CODEFRAME_META_BUILD_ROLE( codeframe::OBJECT );

        public:
            std::string ConstructPatern() const;

                     cSerializable( const std::string& name, cSerializableInterface* parent = NULL );
            virtual ~cSerializable();

            cSerializablePath&       Path();
            cSerializableStorage&    Storage();
            cSerializableSelectable& Selection();
            cSerializableLua&        Script();
            cPropertyManager&        PropertyManager();
            cSerializableChildList&  ChildList();
            cSerializableIdentity&   Identity();

            void        PulseChanged( bool fullTree = false );
            void        CommitChanges();
            void        Enable( bool val );

        private:
            cSerializablePath       m_SerializablePath;
            cSerializableStorage    m_SerializableStorage;
            cSerializableSelectable m_SerializableSelectable;
            cSerializableLua        m_SerializableLua;
            cPropertyManager        m_PropertyManager;
            cSerializableChildList  m_childList;
            cSerializableIdentity   m_Identity;
            WrMutex                 m_Mutex;
    };
}

#endif
