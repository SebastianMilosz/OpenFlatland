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
        friend class PropertyBase;

        public:
            virtual std::string             ObjectName( bool idSuffix = true ) const = 0;   ///< Nazwa serializowanego objektu
            virtual std::string             Class()      const = 0;         ///< Nazwa serializowanej klasy
            virtual std::string             Role()       const = 0;         ///< Rola serializowanego obiektu
            virtual std::string             BuildType()  const = 0;         ///< Sposob budowania obiektu (statycznym, dynamiczny)
            virtual std::string             ConstructPatern() const = 0;    ///< Parametry konstruktora

            virtual void                     SetName( const std::string& name ) = 0;
            virtual bool                     IsNameUnique    ( const std::string& name, bool checkParent = false ) const = 0;

            virtual cSerializablePath&       Path() = 0;
            virtual cSerializableStorage&    Storage() = 0;
            virtual cSerializableSelectable& Selection() = 0;
            virtual cSerializableLua&        Script() = 0;
            virtual cPropertyManager&        PropertyManager() = 0;

            virtual void                     RegisterProperty   ( PropertyBase* prop ) = 0;
            virtual void                     UnRegisterProperty ( PropertyBase* prop ) = 0;
            virtual void                     PulseChanged       ( bool fullTree = false ) = 0;
            virtual void                     CommitChanges      () = 0;
            virtual void                     Enable             ( bool val ) = 0;
            virtual void                     ParentUnbound      () = 0;
            virtual void                     ParentBound        ( cSerializableInterface* obj ) = 0;

            cSerializableChildList* ChildList()       { return &m_childList;}
            void                    Lock     () const { m_Mutex.Lock();     }
            void                    Unlock   () const { m_Mutex.Unlock();   }

            int  GetId() const { return m_Id; }
            void SetId( int id ) { m_Id = id; }

            // Library version nr. and string
            static float       LibraryVersion();
            static std::string LibraryVersionString();

        protected:
                     cSerializableInterface();
            virtual ~cSerializableInterface();

            virtual bool IsPulseState() const = 0;

            mutable WrMutex             m_Mutex;
            cSerializableChildList      m_childList;
            int                         m_Id;
    };

}

#endif // SERIALIZABLEBASE_H_INCLUDED
