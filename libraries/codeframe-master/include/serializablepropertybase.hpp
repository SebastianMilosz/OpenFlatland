#ifndef SERIALIZABLEPROPERTYBASE_HPP_INCLUDED
#define SERIALIZABLEPROPERTYBASE_HPP_INCLUDED

#include "serializableregister.hpp"
#include "serializablepropertyinfo.hpp"
#include "serializable_property_node.hpp"
#include "typeinfo.hpp"
#include "ThreadUtilities.h"

#include <sigslot.h>
#include <climits>
#include <string>

namespace codeframe
{
    class ObjectNode;

    /*****************************************************************************
     * @class PropertyBase
     *****************************************************************************/
    class PropertyBase : public PropertyNode
    {
        friend class cPropertyInfo;

        public:

            PropertyBase( ObjectNode* parentpc, const std::string& name, eType type, cPropertyInfo info ) :
                m_reference(NULL),
                m_referenceParent(NULL),
                m_type(type),
                m_parentpc( parentpc ),
                m_name(name),
                m_id(0),
                m_isWaitForUpdate( false ),
                m_waitForUpdateCnt(0),
                m_propertyInfo( info, this ),
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
            virtual bool operator==(const PropertyBase& sval) const;
            virtual bool operator!=(const PropertyBase& sval) const;

            virtual bool operator==(const int& sval) const;
            virtual bool operator!=(const int& sval) const;

            // Operatory przypisania
            virtual PropertyNode& operator=(const PropertyNode& val);
            virtual PropertyNode& operator=(const bool          val);
            virtual PropertyNode& operator=(const char          val);
            virtual PropertyNode& operator=(const unsigned char val);
            virtual PropertyNode& operator=(const int           val);
            virtual PropertyNode& operator=(const unsigned int  val);
            virtual PropertyNode& operator=(const float         val);
            virtual PropertyNode& operator=(const double        val);
            virtual PropertyNode& operator=(const std::string&  val);
            virtual PropertyNode& operator++();
            virtual PropertyNode& operator--();
            virtual PropertyNode& operator+=(const PropertyNode& rhs);
            virtual PropertyNode& operator-=(const PropertyNode& rhs);
            virtual PropertyNode& operator+ (const PropertyNode& rhs);
            virtual PropertyNode& operator- (const PropertyNode& rhs);
            virtual PropertyNode& operator+=(const int rhs);
            virtual PropertyNode& operator-=(const int rhs);

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
            virtual std::string     ToString();
            int                     ToEnumPosition( const std::string& enumStringValue );
            cPropertyInfo&          Info() { return m_propertyInfo; }
            virtual void            WaitForUpdatePulse();
            virtual void            WaitForUpdate( int time = 100 );
            virtual std::string     Name() const;
            virtual bool            NameIs( const std::string& name ) const;
            virtual uint32_t        Id() const;
            virtual eType           Type() const;
            virtual std::string     Path(bool addName = true) const;
            virtual ObjectNode*     Parent() const { return m_parentpc; }
            virtual PropertyNode*   Reference() const { return m_reference; }
            virtual bool            ConnectReference( smart_ptr<PropertyNode> refNode );
            virtual std::string     TypeString() const;

            virtual std::string     PreviousValueString() const;
            virtual std::string     CurentValueString() const;
            virtual int             PreviousValueInteger() const;
            virtual int             CurentValueInteger() const;

            void                    PulseChanged();
            void                    CommitChanges();
            bool                    IsChanged() const;
            PropertyNode&           WatchdogGetValue( int time = 1000 );

            virtual void            SetNumber( const int val );
            virtual int             GetNumber() const;
            virtual void            SetReal( const double val );
            virtual double          GetReal() const;
            virtual void            SetString( const std::string&  val );
            virtual std::string     GetString() const;

            virtual void Lock() const
            {
                m_Mutex.Lock();
            }

            virtual void Unlock() const
            {
                m_Mutex.Unlock();
            }

        protected:
            static int      s_globalParConCnt;
            PropertyNode*   m_reference;             ///< Wskaznik na sprzezone z tym polem pole
            ObjectNode*     m_referenceParent;
            eType           m_type;
            ObjectNode*     m_parentpc;
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
