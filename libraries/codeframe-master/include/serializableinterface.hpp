#ifndef SERIALIZABLEBASE_H_INCLUDED
#define SERIALIZABLEBASE_H_INCLUDED

#include <vector>
#include <string>
#include <map>
#include <typeinfo>
#include <smartpointer.h>
#include <sigslot.h>

#include <DataTypesUtilities.h>
#include <MathUtilities.h>
#include <ThreadUtilities.h>

#include "serializableproperty.hpp"
#include "serializablechildlist.hpp"
#include "serializableidentity.hpp"
#include "serializablepath.hpp"
#include "serializablestorage.hpp"
#include "serializableselectable.hpp"
#include "serializablepropertymanager.hpp"
#include "serializablelua.hpp"
#include "xmlformatter.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief Base common Interface to access to all cSerializable objects
      * @author Sebastian Milosz
      * @version 1.0
      * @note Base common Interface to access to all cSerializable objects
     **
    ******************************************************************************/
    class cSerializableInterface : public sigslot::has_slots<>
    {
        public:
            virtual std::string Class()             const = 0;    ///< Class name meta data
            virtual std::string ConstructPatern()   const = 0;    ///< Constructor parameters patern
            virtual eBuildRole  Role()              const = 0;    ///< Class role meta data
            virtual eBuildType  BuildType()         const = 0;    ///< Class build type meta data

            virtual cSerializablePath&       Path() = 0;
            virtual cSerializableStorage&    Storage() = 0;
            virtual cSerializableSelectable& Selection() = 0;
            virtual cSerializableLua&        Script() = 0;
            virtual cPropertyManager&        PropertyManager() = 0;
            virtual cSerializableChildList&  ChildList() = 0;
            virtual cSerializableIdentity&   Identity() = 0;

            virtual void PulseChanged( bool fullTree = false ) = 0;
            virtual void CommitChanges() = 0;
            virtual void Enable( bool val ) = 0;

        protected:
                     cSerializableInterface();
            virtual ~cSerializableInterface();
    };

}

#endif // SERIALIZABLEBASE_H_INCLUDED
