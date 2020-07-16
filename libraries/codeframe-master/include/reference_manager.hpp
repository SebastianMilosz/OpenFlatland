#ifndef REFERENCE_MANAGER_HPP_INCLUDED
#define REFERENCE_MANAGER_HPP_INCLUDED

#include <map>
#include <string>
#include <smartpointer.h>

#include "serializable_property_node.hpp"
#include "serializable_object_selection.hpp"

namespace codeframe
{
    class PropertyBase;
    class ObjectNode;

    class ReferenceManager
    {
        public:
            class Inhibit
            {
                public:
                    Inhibit(ObjectNode& node) :
                        m_node(node)
                    {
                        ReferenceManager::m_inhibitResolveReferences = true;
                    }
                   ~Inhibit()
                    {
                        ReferenceManager::m_inhibitResolveReferences = false;
                        ReferenceManager::ResolveReferences(m_node);
                    }
                private:
                    ObjectNode& m_node;
            };

            ReferenceManager();
           ~ReferenceManager() = default;

            void SetReference( const std::string& refPath, PropertyBase* prop = NULL );
            void SetProperty( PropertyBase* prop );
            const std::string& Get() const;

            static void ResolveReferences( ObjectNode& root );
            static void LogUnresolvedReferences();
            static unsigned int UnresolvedReferencesCount();
        private:
            friend class Inhibit;
            struct sReferenceData
            {
                smart_ptr<PropertyNode> Property;
                std::string RefPath;
            };

            static bool_t m_inhibitResolveReferences;
            std::string m_referencePath;
            smart_ptr<PropertyNode> m_property;
            static std::map<PropertyNode*, sReferenceData> m_referencePathMap;
    };
}

#endif // REFERENCE_MANAGER_HPP_INCLUDED
