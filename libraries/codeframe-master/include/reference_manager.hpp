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
            ReferenceManager();
            ~ReferenceManager();

            void SetReference( const std::string& refPath, PropertyBase* prop = NULL );
            void SetProperty( PropertyBase* prop );
            const std::string& Get() const;

            static void ResolveReferences( ObjectNode& root );

            static void LogUnresolvedReferences();
        private:
            struct sReferenceData
            {
                smart_ptr<PropertyNode> Property;
                std::string RefPath;
            };

            static std::string PreparePath(const std::string& path, smart_ptr<ObjectSelection> propertyParent);
            static bool_t IsDownHierarchy(const std::string& path);

            std::string m_referencePath;
            smart_ptr<PropertyNode> m_property;
            static std::map<PropertyNode*, sReferenceData> m_referencePathMap;
    };
}

#endif // REFERENCE_MANAGER_HPP_INCLUDED
