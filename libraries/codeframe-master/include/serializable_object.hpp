#ifndef _CSERIALIZABLE_OBJECT_H
#define _CSERIALIZABLE_OBJECT_H

#include "serializable_object_node.hpp"
#include "serializable_property_base.hpp"
#include "serializable_storage.hpp"

#define CODEFRAME_META_CLASS_NAME(p) public: std::string Class() const override { return p; }
#define CODEFRAME_META_CONSTRUCT_PATERN(p) public: std::string ConstructPatern() const override { return p; }
#define CODEFRAME_META_BUILD_ROLE(p) public: codeframe::eBuildRole Role() const override { return p; }
#define CODEFRAME_META_BUILD_TYPE(p) public: codeframe::eBuildType BuildType() const override { return p; }

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

                     Object( const std::string& name, ObjectNode* parent = nullptr );
                     Object( const std::string& name, smart_ptr<ObjectNode> parent );
            virtual ~Object();

            cPath&         Path() override;
            cStorage&      Storage() override;
            cSelectable&   Selection() override;
            cScript&       Script() override;
            cPropertyList& PropertyList() override;
            cObjectList&   ChildList() override;
            cIdentity&     Identity() override;

            smart_ptr<ObjectSelection> Create(
                                            const std::string& className,
                                            const std::string& objName,
                                            const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                         ) override;

            unsigned int Count() const override;

            smart_ptr<ObjectSelection> operator[]( const unsigned int i ) override;
            smart_ptr<ObjectSelection> operator[]( const std::string& name ) override;

            smart_ptr<ObjectSelection> Child( const unsigned int i ) override;
            smart_ptr<ObjectSelection> Child( const std::string& name ) override;

            smart_ptr<PropertyNode> Property(const std::string& name) override;
            smart_ptr<PropertyNode> PropertyFromPath(const std::string& path) override;

            std::string ObjectName( bool idSuffix = true ) const override;

            void PulseChanged( bool fullTree = false ) override;
            void CommitChanges() override;
            void Enable( bool val ) override;

        private:
            cPath         m_SerializablePath;
            cStorage      m_SerializableStorage;
            cSelectable   m_SerializableSelectable;
            cScript       m_SerializableScript;
            cPropertyList m_PropertyList;
            cObjectList   m_childList;
            cIdentity     m_Identity;
            WrMutex       m_Mutex;
    };
}

#endif // _CSERIALIZABLE_OBJECT_H
