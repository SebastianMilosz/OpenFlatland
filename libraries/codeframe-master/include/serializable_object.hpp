#ifndef _CSERIALIZABLE_OBJECT_H
#define _CSERIALIZABLE_OBJECT_H

#include "serializable_object_node.hpp"
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
      * @version 1.0
      * @note Object
     **
    ******************************************************************************/
    class Object : public ObjectNode
    {
        CODEFRAME_META_BUILD_ROLE( codeframe::OBJECT );

        public:
            std::string ConstructPatern() const;

                     Object( const std::string& name, ObjectNode* parent = NULL );
            virtual ~Object();

            cSerializablePath&       Path();
            cSerializableStorage&    Storage();
            cSerializableSelectable& Selection();
            cSerializableScript&     Script();
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
            cSerializableScript     m_SerializableScript;
            cPropertyManager        m_PropertyManager;
            cSerializableChildList  m_childList;
            cSerializableIdentity   m_Identity;
            WrMutex                 m_Mutex;
    };
}

#endif // _CSERIALIZABLE_OBJECT_H