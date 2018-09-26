#ifndef SERIALIZABLEPROPERTYBASE_HPP_INCLUDED
#define SERIALIZABLEPROPERTYBASE_HPP_INCLUDED

#include "serializableregister.hpp"
#include "serializablepropertyinfo.hpp"
#include "typeinfo.hpp"
#include "ThreadUtilities.h"

#include <sigslot.h>
#include <climits>
#include <string>

namespace codeframe
{
    class cSerializable;

    /*****************************************************************************
     * @class PropertyBase
     *****************************************************************************/
    class PropertyBase
    {
        friend class cPropertyInfo;

        public:

            PropertyBase( cSerializableInterface* parentpc, const std::string& name, eType type, cPropertyInfo info ) :
                m_reference(NULL),
                m_referenceParent(NULL),
                m_type(type),
                m_parentpc( parentpc ),
                m_name(name),
                m_id(0),
                m_isWaitForUpdate( false ),
                m_waitForUpdateCnt(0),
                m_propertyInfo( info, parentpc ),
                m_pulseAbort( false ),
                m_temporary( false )
                {
                    RegisterProperty();
                }

            virtual ~PropertyBase()
            {
                if( m_temporary == false )
                {
                    UnRegisterProperty();
                }
            }

            // Sygnaly
            sigslot::signal1<PropertyBase*> signalChanged;

            // Copy operator
            PropertyBase( const PropertyBase& sval ) :
                m_reference      (sval.m_reference),
                m_referenceParent(sval.m_referenceParent),
                m_type           (sval.m_type),
                m_parentpc       (sval.m_parentpc),
                m_name           (sval.m_name),
                m_id             (sval.m_id),
                m_propertyInfo   (sval.m_propertyInfo),
                m_pulseAbort     (sval.m_pulseAbort),
                m_temporary      ( true )
            {
            }

            // Operator porownania
            virtual bool operator==(const PropertyBase& sval);
            virtual bool operator!=(const PropertyBase& sval);

            virtual bool operator==(const int& sval);
            virtual bool operator!=(const int& sval);

            // Operatory przypisania
            virtual PropertyBase& operator=(const PropertyBase& val);
            virtual PropertyBase& operator=(const bool          val);
            virtual PropertyBase& operator=(const char          val);
            virtual PropertyBase& operator=(const unsigned char val);
            virtual PropertyBase& operator=(const int           val);
            virtual PropertyBase& operator=(const unsigned int  val);
            virtual PropertyBase& operator=(const float         val);
            virtual PropertyBase& operator=(const double        val);
            virtual PropertyBase& operator=(const std::string&  val);
            virtual PropertyBase& operator++();
            virtual PropertyBase& operator--();
            virtual PropertyBase& operator+=(const PropertyBase& rhs);
            virtual PropertyBase& operator-=(const PropertyBase& rhs);
            virtual PropertyBase  operator+(const PropertyBase& rhs);
            virtual PropertyBase  operator-(const PropertyBase& rhs);
            virtual PropertyBase& operator+=(const int rhs);
            virtual PropertyBase& operator-=(const int rhs);

            // Operatory rzutowania
            virtual operator bool() const;
            virtual operator char() const;
            virtual operator unsigned char() const;
            virtual operator int() const;
            virtual operator unsigned int() const;
            virtual operator unsigned short() const;
            virtual operator double() const;
            virtual operator float() const;
            virtual operator std::string() const;

            bool                    IsReference() const;
            int                     ToInt() const { return (int)(*this); }
            std::string             ToString();
            int                     ToEnumPosition( const std::string& enumStringValue );
            cPropertyInfo&          Info() { return m_propertyInfo; }
            virtual void            WaitForUpdatePulse();
            virtual void            WaitForUpdate( int time = 100 );
            virtual std::string     Name() const;
            virtual bool            NameIs( const std::string& name ) const;
            virtual uint32_t        Id() const;
            virtual eType           Type() const;
            virtual std::string     Path(bool addName = true) const;
            virtual cSerializableInterface*  Parent() { return m_parentpc; }
            virtual PropertyBase*   Reference() { return m_reference; }
            virtual bool            ConnectReference( PropertyBase* refProp );
            virtual std::string     TypeString() const;

            virtual std::string     PreviousValueString() const;
            virtual std::string     CurentValueString() const;
            virtual int             PreviousValueInteger() const;
            virtual int             CurentValueInteger() const;

            void                    PulseChanged();
            void                    CommitChanges();
            bool                    IsChanged() const;
            PropertyBase&           WatchdogGetValue( int time = 1000 );

            void                    SetNumber( int val );
            int                     GetNumber() const;
            void                    SetReal( double val );
            double                  GetReal() const;
            void                    SetString( const std::string&  val );
            std::string             GetString() const;

        protected:
            static int      s_globalParConCnt;
            PropertyBase*   m_reference;             ///< Wskaznik na sprzezone z tym polem pole
            cSerializableInterface*  m_referenceParent;
            eType           m_type;
            cSerializableInterface*  m_parentpc;
            std::string     m_name;
            uint32_t        m_id;
            mutable WrMutex m_Mutex;
            bool            m_isWaitForUpdate;
            int             m_waitForUpdateCnt;
            cPropertyInfo   m_propertyInfo;
            bool            m_pulseAbort;
            bool            m_temporary;

            void     RegisterProperty();
            void     UnRegisterProperty();

            static uint32_t GetHashId( const std::string& str, uint16_t mod = 0 );
    };
}

#endif // SERIALIZABLEPROPERTYBASE_HPP_INCLUDED
