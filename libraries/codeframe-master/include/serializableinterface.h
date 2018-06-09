#ifndef SERIALIZABLEBASE_H_INCLUDED
#define SERIALIZABLEBASE_H_INCLUDED

#include "serializableproperty.h"
#include "serializablechildlist.h"

#include <vector>
#include <string>
#include <map>
#include <xmlformatter.h>
#include <typeinfo>
#include <sigslot.h>

#include <DataTypesUtilities.h>
#include <MathUtilities.h>
#include <ThreadUtilities.h>

namespace codeframe
{

    /*****************************************************************************/
    /**
      * @brief Base common Interface to acces to all cSerializable objects
      * @author Sebastian Milosz
      * @version 1.0
      * @note Base common Interface to acces to all cSerializable objects
     **
    ******************************************************************************/
    class cSerializableInterface : public sigslot::has_slots<>
    {
        public:
            class iterator : public std::iterator<std::input_iterator_tag, Property*>
            {
                friend class Property;
                friend class Property_Int;
                friend class Property_Str;
                friend class cSerializableInterface;

            public:
                // Konstruktor kopiujacy
                iterator(const iterator& n) : m_base(n.m_base), m_param(n.m_param), m_curId(n.m_curId) {}

                // Operator wskaznikowy wyodrebnienia wskazywanej wartosci
                Property* operator *()
                {
                    m_param = m_base->GetObjectFieldValue( m_curId );
                    return m_param;
                }

                // Operator inkrementacji (przejscia na kolejne pole)
                iterator& operator ++(){ if(m_curId < m_base->GetObjectFieldCnt()) ++m_curId; return *this; }

                // Operatory porownania
                bool operator< (const iterator& n) { return   n.m_curId <  m_curId;  }
                bool operator> (const iterator& n) { return   n.m_curId >  m_curId;  }
                bool operator<=(const iterator& n) { return !(n.m_curId >  m_curId); }
                bool operator>=(const iterator& n) { return !(n.m_curId <  m_curId); }
                bool operator==(const iterator& n) { return   n.m_curId == m_curId;  }
                bool operator!=(const iterator& n) { return !(n.m_curId == m_curId); }

            private:
                // Konstruktor bazowy prywatny bo tylko cSerializable moze tworzyc swoje iteratory
                iterator(cSerializableInterface* b, int n) : m_base(b), m_curId(n) {}

            private:
                   cSerializableInterface* m_base;
                   Property*               m_param;
                   int                     m_curId;
            };

        public:
            virtual std::string             ObjectName() const = 0;   ///< Nazwa serializowanego objektu
            virtual std::string             Class()      const = 0;   ///< Nazwa serializowanej klasy
            virtual std::string             Role()       const = 0;   ///< Rola serializowanego obiektu
            virtual std::string             BuildType()  const = 0;   ///< Sposob budowania obiektu (statycznym, dynamiczny)
            virtual bool                    IsPropertyUnique( std::string const& name ) const = 0;
            virtual bool                    IsNameUnique    ( std::string const& name, bool checkParent = false ) const = 0;
            virtual std::string             Path() const = 0;
            virtual cSerializableInterface* GetRootObject() = 0;
            virtual Property*               GetPropertyByName  ( std::string const& name ) = 0;
            virtual Property*               GetPropertyById    ( uint32_t    id   ) = 0;
            virtual Property*               GetPropertyFromPath( std::string const& path ) = 0;
            virtual cSerializableInterface* GetChildByName     ( std::string const& name ) = 0;
            virtual void                    PulseChanged       ( bool fullTree = false ) = 0;
            virtual void                    CommitChanges      () = 0;
            virtual void                    Enable             ( bool val ) = 0;
            virtual void                    ParentUnbound      () = 0;
            virtual void                    ParentBound        ( cSerializableInterface* obj ) = 0;

            cSerializableChildList* ChildList()       { return &m_childList;}
            void                    Lock     () const { m_Mutex.Lock();     }
            void                    Unlock   () const { m_Mutex.Unlock();   }

            // Library version nr. and string
            static float       LibraryVersion();
            static std::string LibraryVersionString();

            // Iterator
            iterator begin() throw();
            iterator end()   throw();
            int      size()  const;

        protected:
                     cSerializableInterface();
            virtual ~cSerializableInterface();

            Property* GetObjectFieldValue( int cnt );     ///< Zwraca wartosc pola do serializacji
            int       GetObjectFieldCnt() const;          ///< Zwraca ilosc skladowych do serializacji

            std::vector<Property*>  m_vMainPropertyList;  ///< Kontenet zawierajacy wskazniki do parametrow
            Property_Int            m_dummyProperty;
            mutable WrMutex         m_Mutex;
            cSerializableChildList  m_childList;
    };

}

#endif // SERIALIZABLEBASE_H_INCLUDED
