#ifndef OBJECT_NODE_H_INCLUDED
#define OBJECT_NODE_H_INCLUDED

#include <vector>
#include <array>
#include <string>
#include <map>
#include <typeinfo>
#include <smartpointer.h>
#include <sigslot.h>

#include <DataTypesUtilities.h>
#include <MathUtilities.h>
#include <ThreadUtilities.h>

#include "serializable_property.hpp"
#include "serializable_object_list.hpp"
#include "serializable_identity.hpp"
#include "serializable_path.hpp"
#include "serializable_storage.hpp"
#include "serializable_selectable.hpp"
#include "serializable_property_list.hpp"
#include "serializable_lua.hpp"
#include "xmlformatter.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief Base common Interface to access to all codeframe Objects
      * @version 1.0
      * @note Base common Interface to access to all codeframe Objects
     **
    ******************************************************************************/
    class ObjectNode : public sigslot::has_slots<>
    {
        public:
            virtual std::string Class()           const = 0;    ///< Class name meta data
            virtual std::string ConstructPatern() const = 0;    ///< Constructor parameters patern
            virtual eBuildRole  Role()            const = 0;    ///< Class role meta data
            virtual eBuildType  BuildType()       const = 0;    ///< Class build type meta data

            virtual ~ObjectNode();

            virtual cPath&          Path() = 0;
            virtual cStorage&       Storage() = 0;
            virtual cSelectable&    Selection() = 0;
            virtual cScript&        Script() = 0;
            virtual cPropertyList&  PropertyList() = 0;
            virtual cObjectList&    ChildList() = 0;
            virtual cIdentity&      Identity() = 0;

            virtual const std::vector< std::string >& ClassSet() const
            {
                static std::vector< std::string > tmp;
                return tmp;
            }

            virtual const std::vector< std::vector<std::string> >& ClassParameterSet() const
            {
                static std::vector< std::vector<std::string> > tmp;
                return tmp;
            }

            virtual smart_ptr<codeframe::Object> Create(
                                                  const std::string& className,
                                                  const std::string& objName,
                                                  const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                 ) = 0;

            virtual unsigned int Count() const = 0;

            virtual smart_ptr<ObjectSelection> operator[]( const unsigned int i ) = 0;
            virtual smart_ptr<ObjectSelection> operator[]( const std::string& name ) = 0;

            virtual smart_ptr<ObjectSelection> Child( const unsigned int i ) = 0;
            virtual smart_ptr<ObjectSelection> Child( const std::string& name ) = 0;

            virtual smart_ptr<PropertyNode> Property(const std::string& name) = 0;
            virtual smart_ptr<PropertyNode> PropertyFromPath(const std::string& path) = 0;

            virtual std::string ObjectName( bool idSuffix = true ) const = 0;

            virtual smart_ptr<ObjectSelection> Parent() const = 0;
            virtual smart_ptr<ObjectSelection> Root() = 0;
            virtual smart_ptr<ObjectSelection> ObjectFromPath( const std::string& path ) = 0;
            virtual smart_ptr<ObjectSelection> GetObjectByName( const std::string& name ) = 0;
            virtual smart_ptr<ObjectSelection> GetObjectById( const uint32_t id ) = 0;

            virtual void PulseChanged( bool fullTree = false ) = 0;
            virtual void CommitChanges() = 0;
            virtual void Enable( bool val ) = 0;

            virtual void Unbound() = 0;

            // Signals
            sigslot::signal1<void*> signalDeleted;
        protected:
            ObjectNode();
    };

}

#endif // OBJECT_NODE_H_INCLUDED
