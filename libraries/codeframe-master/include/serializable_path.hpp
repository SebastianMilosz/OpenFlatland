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
                    void PathPushBack(const std::string& val);
                    size_t size() const noexcept;
                    std::string at(size_t pos);
                    void reverse();
                    void FromDirString(const std::string& val);
                    std::string ToDirString() const;
                    operator std::string() const;

                private:
                    std::vector<std::string> m_ObjectPath;
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

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline void cPath::sPathLink::PathPushBack(const std::string& val)
    {
        m_ObjectPath.push_back(val);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline size_t cPath::sPathLink::size() const noexcept
    {
        return m_ObjectPath.size();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline std::string cPath::sPathLink::at(size_t pos)
    {
        return m_ObjectPath.at(pos);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline void cPath::sPathLink::reverse()
    {
        std::reverse(std::begin(m_ObjectPath), std::end(m_ObjectPath));
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline void cPath::sPathLink::FromDirString(const std::string& val)
    {
        utilities::text::split( val, m_delimiters, m_ObjectPath);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline std::string cPath::sPathLink::ToDirString() const
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

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline cPath::sPathLink::operator std::string() const
    {
        std::string retLine("vector[");
        for (auto& n : m_ObjectPath)
        {
            retLine += (std::string)n + ",";
        }
        retLine += "]";
        return retLine;
    }
}

#endif // CPATH_HPP_INCLUDED
