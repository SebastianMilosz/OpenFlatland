#ifndef CPATH_HPP_INCLUDED
#define CPATH_HPP_INCLUDED

#include <string>
#include <smartpointer.h>
#include <typedefs.hpp>
#include <algorithm>    // std::reverse
#include <TextUtilities.h>

#include "serializable_property_node.hpp"
#include "serializable_object_selection.hpp"
#include "serializable_object_multiple_selection.hpp"

namespace codeframe
{
    class ObjectNode;

    class cPath
    {
        public:
             cPath( ObjectNode& sint );
            ~cPath();

            struct sPathLink
            {
                public:
                    void SetPropertyName(const std::string& val)
                    {
                        m_PropertyName = val;
                    }

                    void PathPushBack(const std::string& val)
                    {
                        m_ObjectPath.push_back(val);
                    }

                    size_t size() const noexcept
                    {
                        return m_ObjectPath.size();
                    }

                    std::string at(size_t pos)
                    {
                        return m_ObjectPath.at(pos);
                    }

                    void reverse()
                    {
                        std::reverse(std::begin(m_ObjectPath), std::end(m_ObjectPath));
                    }

                    void FromDirString(const std::string& val)
                    {
                        utilities::text::split( val, m_delimiters, m_ObjectPath);
                    }

                    std::string ToDirString() const
                    {
                        std::string dirString;
                        for (auto it = m_ObjectPath.begin(); it != m_ObjectPath.end(); ++it)
                        {
                            dirString += *it;
                            dirString += "/";
                        }
                        dirString.pop_back();   // Remove last / char
                        return dirString;
                    }

                    operator std::string() const
                    {
                        std::string retLine("vector[");
                        for (auto& n : m_ObjectPath)
                        {
                            retLine += (std::string)n + ",";
                        }
                        retLine += "]";
                        return retLine;
                    }

                private:
                    std::string              m_PropertyName;
                    std::vector<std::string> m_ObjectPath;
                    smart_ptr<PropertyNode>  m_Property;
                    smart_ptr<ObjectNode>    m_Object;
            };

            std::string PathString() const;
            bool_t ParentBound( smart_ptr<ObjectNode> parent );
            void ParentUnbound();

            bool_t IsNameUnique( const std::string& name, const bool_t checkParent = false ) const;

            smart_ptr<ObjectSelection> Parent() const;
            smart_ptr<ObjectSelection> GetRootObject();
            smart_ptr<ObjectSelection> GetObjectFromPath( const std::string& path );

            static void PreparePathLink(const std::string& pathString, cPath::sPathLink& pathLink, smart_ptr<ObjectSelection> propertyParent);

        private:
            static const std::string m_delimiters;
            static bool_t IsDownHierarchy(const std::string& path);
            static bool_t IsRelativeHierarchy(const std::string& path);

            ObjectNode& m_sint;
            smart_ptr<ObjectNode> m_parent;
    };

}

#endif // CPATH_HPP_INCLUDED
