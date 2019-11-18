#ifndef OBJECT_NODE_H_INCLUDED
#define OBJECT_NODE_H_INCLUDED

#include <vector>
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

            virtual cPath&          Path() = 0;
            virtual cStorage&       Storage() = 0;
            virtual cSelectable&    Selection() = 0;
            virtual cScript&        Script() = 0;
            virtual cPropertyList&  PropertyList() = 0;
            virtual cObjectList&    ChildList() = 0;
            virtual cIdentity&      Identity() = 0;

            virtual void PulseChanged( bool fullTree = false ) = 0;
            virtual void CommitChanges() = 0;
            virtual void Enable( bool val ) = 0;

        protected:
                     ObjectNode();
            virtual ~ObjectNode();
    };

}

#endif // OBJECT_NODE_H_INCLUDED
