#ifndef _SERIALIZABLEPROPERTY_H
#define _SERIALIZABLEPROPERTY_H

#include "serializableregister.h"
#include "serializablepropertyinfo.hpp"
#include "typeinfo.hpp"

#include <ThreadUtilities.h>

#include <sigslot.h>
#include <climits>
#include <string>

#ifdef SERIALIZABLE_USE_CIMAGE
#include <cimage.h>
#endif

namespace codeframe
{
    class cSerializable;

    /*****************************************************************************
     * @class Property
     *****************************************************************************/
    class PropertyBase
    {
        friend class cPropertyInfo;

        public:

            PropertyBase( cSerializable* parentpc, std::string name, eType type, cPropertyInfo info ) :
                m_reference(NULL),
                m_referenceParent(NULL),
                m_type(type),
                m_parentpc(parentpc),
                m_name(name),
                m_id(0),
                m_isWaitForUpdate(false),
                m_waitForUpdateCnt(0),
                m_propertyInfo( info ),
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
            virtual PropertyBase& operator=(PropertyBase  val);
            virtual PropertyBase& operator=(bool          val);
            virtual PropertyBase& operator=(char          val);
            virtual PropertyBase& operator=(unsigned char val);
            virtual PropertyBase& operator=(int           val);
            virtual PropertyBase& operator=(unsigned int  val);
            virtual PropertyBase& operator=(float         val);
            virtual PropertyBase& operator=(double        val);
            virtual PropertyBase& operator=(std::string   val);
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
            virtual operator int() const;
            virtual operator unsigned int() const;
            virtual operator unsigned short() const;
            virtual operator double() const;
            virtual operator std::string() const;

            bool                    IsReference() const;
            int                     ToInt() const { return (int)(*this); }
            std::string             ToString();
            int                     ToEnumPosition( std::string enumStringValue );
            cPropertyInfo&          Info() { return m_propertyInfo; }
            virtual void        	WaitForUpdatePulse();
            virtual void         	WaitForUpdate(int time = 100);
            virtual std::string  	Name() const;
            virtual bool  	        NameIs( std::string name ) const;
            virtual uint32_t     	Id() const;
            virtual eType        	Type() const;
            virtual std::string  	Path(bool addName = true) const;
            virtual cSerializable* 	Parent() { return m_parentpc; }
            virtual PropertyBase*   Reference() { return m_reference; }
            virtual bool         	ConnectReference( PropertyBase* refProp );
            virtual std::string  	TypeString() const;

            virtual std::string  	PreviousValueString() const;
            virtual std::string  	CurentValueString() const;
            virtual int             PreviousValueInteger() const;
            virtual int             CurentValueInteger() const;

            void                 	PulseChanged();
            void                    CommitChanges();
            bool                    IsChanged() const;
            PropertyBase&           WatchdogGetValue( int time = 1000 );

            void         	        SetNumber( int val );
            int                     GetNumber() const;
            void         	        SetReal( double val );
            double                  GetReal() const;
            void         	        SetString( std::string  val );
            std::string             GetString() const;

        protected:
            static int      s_globalParConCnt;
            PropertyBase*   m_reference;             ///< Wskaznik na sprzezone z tym polem pole
            cSerializable*  m_referenceParent;
            eType           m_type;
            cSerializable*  m_parentpc;
            std::string     m_name;
            uint32_t        m_id;
            mutable WrMutex m_Mutex;
            bool            m_isWaitForUpdate;
            int             m_waitForUpdateCnt;
            cPropertyInfo   m_propertyInfo;
            bool            m_pulseAbort;
            bool            m_temporary;

            void     RegisterProperty();
            void	 UnRegisterProperty();
            void     ValueUpdate();

            static uint32_t GetHashId(std::string str, uint16_t mod = 0 );
    };

    //
    template <typename retT, typename classT = cSerializable >
    class Property : public PropertyBase
    {
        public:
            Property( cSerializable* parentpc,
                      std::string name,
                      retT val,
                      cPropertyInfo info,
                      retT (classT::*getValue)() = NULL,
                      void (classT::*setValue)(retT) = NULL
                     ) : PropertyBase( parentpc, name, GetTypeInfo<retT>().TypeCode, info )
            {
                GetValueCallback = getValue;
                SetValueCallback = setValue;
                m_baseValue      = val;
            }

            // Destructor
            virtual ~Property()
            {

            }

            // Copy operator
            Property( const Property& sval ) : PropertyBase( sval )
            {
                // Values
                m_baseValue = sval.m_baseValue;
                m_baseValuePrew = sval.m_baseValuePrew;

                // Funktors
                GetValueCallback = sval.GetValueCallback;
                SetValueCallback = sval.SetValueCallback;
            }

            // Comparison operators
            virtual bool operator==(const Property& sval)
            {
                m_Mutex.Lock();
                bool retVal = false;
                if ( m_baseValue == sval.m_baseValue)
                {
                    retVal = true;
                }
                m_Mutex.Unlock();

                return retVal;
            }

            // Comparison operators
            virtual bool operator!=(const Property& sval)
            {
                m_Mutex.Lock();
                bool retVal = false;
                if ( m_baseValue != sval.m_baseValue)
                {
                    retVal = true;
                }
                m_Mutex.Unlock();

                return retVal;
            }

            // Comparison operators
            virtual bool operator==(const int& sval)
            {
                bool retVal = false;

                m_Mutex.Lock();
                int thisValue = GetTypeInfo<retT>().ToInteger( (void*)&m_baseValue );

                if( thisValue == sval )
                {
                    retVal = true;
                }

                m_Mutex.Unlock();

                return retVal;
            }

            virtual bool operator!=(const int& sval)
            {
                return !(*this==sval);
            }

            // Copy operator
            virtual Property& operator=(Property val)
            {
                this->PropertyBase::operator=(val);

                // Values
                m_baseValue = val.m_baseValue;
                m_baseValuePrew = val.m_baseValuePrew;

                // Funktors
                GetValueCallback = val.GetValueCallback;
                SetValueCallback = val.SetValueCallback;

                return *this;
            }

            // From fundamental type bool
            virtual Property& operator=(bool val)
            {
                if( Info().GetEnable() == true )
                {
                    retT* valueT = static_cast<retT*>( GetTypeInfo<retT>().FromInteger( val ) );
                    if( NULL != valueT )
                    {
                        m_Mutex.Lock();
                        m_baseValuePrew = m_baseValue;
                        m_baseValue = *valueT;
                        m_Mutex.Unlock();
                    }
                }
                return *this;
            }

            // From fundamental type char
            virtual Property& operator=(char val)
            {
                if( Info().GetEnable() == true )
                {
                    retT* valueT = static_cast<retT*>( GetTypeInfo<retT>().FromInteger( val ) );
                    if( NULL != valueT )
                    {
                        m_Mutex.Lock();
                        m_baseValuePrew = m_baseValue;
                        m_baseValue = *valueT;
                        m_Mutex.Unlock();
                    }
                }
                return *this;
            }

            // From fundamental type unsigned char
            virtual Property& operator=(unsigned char val)
            {
                if( Info().GetEnable() == true )
                {
                    retT* valueT = static_cast<retT*>( GetTypeInfo<retT>().FromInteger( val ) );
                    if( NULL != valueT )
                    {
                        m_Mutex.Lock();
                        m_baseValuePrew = m_baseValue;
                        m_baseValue = *valueT;
                        m_Mutex.Unlock();
                    }
                }
                return *this;
            }

            // From fundamental type int
            virtual Property& operator=(int val)
            {
                if( Info().GetEnable() == true )
                {
                    retT* valueT = static_cast<retT*>( GetTypeInfo<retT>().FromInteger( val ) );
                    if( NULL != valueT )
                    {
                        m_Mutex.Lock();
                        m_baseValuePrew = m_baseValue;
                        m_baseValue = *valueT;
                        m_Mutex.Unlock();
                    }
                }
                return *this;
            }

            // From fundamental type unsigned int
            virtual Property& operator=(unsigned int val)
            {
                if( Info().GetEnable() == true )
                {
                    retT* valueT = static_cast<retT*>( GetTypeInfo<retT>().FromInteger( val ) );
                    if( NULL != valueT )
                    {
                        m_Mutex.Lock();
                        m_baseValuePrew = m_baseValue;
                        m_baseValue = *valueT;
                        m_Mutex.Unlock();
                    }
                }
                return *this;
            }

            // From fundamental type float
            virtual Property& operator=(float val)
            {
                if( Info().GetEnable() == true )
                {
                    retT* valueT = static_cast<retT*>( GetTypeInfo<retT>().FromReal( val ) );
                    if( NULL != valueT )
                    {
                        m_Mutex.Lock();
                        m_baseValuePrew = m_baseValue;
                        m_baseValue = *valueT;
                        m_Mutex.Unlock();
                    }
                }
                return *this;
            }

            // From fundamental type double
            virtual Property& operator=(double val)
            {
                if( Info().GetEnable() == true )
                {
                    retT* valueT = static_cast<retT*>( GetTypeInfo<retT>().FromReal( val ) );
                    if( NULL != valueT )
                    {
                        m_Mutex.Lock();
                        m_baseValuePrew = m_baseValue;
                        m_baseValue = *valueT;
                        m_Mutex.Unlock();
                    }
                }
                return *this;
            }

            // From extended type std::string
            virtual Property& operator=(std::string val)
            {
                if( Info().GetEnable() == true )
                {
                    retT* valueT = static_cast<retT*>( GetTypeInfo<retT>().FromText( val ) );
                    if( NULL != valueT )
                    {
                        m_Mutex.Lock();
                        m_baseValuePrew = m_baseValue;
                        m_baseValue = *valueT;
                        m_Mutex.Unlock();
                    }
                }
                return *this;
            }

            // ++
            virtual Property& operator++()
            {
                (*this) = (int)(*this) + 1;

                // actual increment takes place here
                return *this;
            }

            // --
            virtual Property& operator--()
            {
                (*this) = (int)(*this) - 1;

                // actual decrement takes place here
                return *this;
            }

            // +=
            virtual Property& operator+=(const Property& rhs)
            {
                *this = *this + rhs;
                return *this;
            }

            // -=
            virtual Property& operator-=(const Property& rhs)
            {
                *this = *this - rhs;
                return *this;
            }

            // +
            virtual Property  operator+(const Property& rhs)
            {
                m_Mutex.Lock();
                m_baseValuePrew = m_baseValue;
                m_baseValue = m_baseValue + rhs.m_baseValue;
                m_Mutex.Unlock();

                return *this;
            }

            // -
            virtual Property  operator-(const Property& rhs)
            {
                m_Mutex.Lock();
                m_baseValuePrew = m_baseValue;
                m_baseValue = m_baseValue - rhs.m_baseValue;
                m_Mutex.Unlock();

                return *this;
            }

            // +=
            virtual Property& operator+=(const int rhs)
            {
                Property<retT, classT> prop(*this);
                prop = rhs;

                *this = *this + prop;
                return *this;
            }

            // -=
            virtual Property& operator-=(const int rhs)
            {
                Property<retT, classT> prop(*this);
                prop = rhs;

                *this = *this - prop;
                return *this;
            }

            //
            virtual Property  operator++(int)
            {
                Property<retT, classT> prop(*this); // copy
                operator++();                       // pre-increment
                return prop;                        // return old value
            }

            //
            virtual Property  operator--(int)
            {
                Property<retT, classT> prop(*this); // copy
                operator--();                       // pre-decrement
                return prop;                        // return old value
            }

            // Fundamental types casting operators
            virtual operator bool() const
            {
                if( m_reference )
                {
                    return (bool)(*m_reference);
                }

                bool retVal = false;

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToInteger( (void*)&m_baseValue );
                m_Mutex.Unlock();

                return retVal;
            }

            virtual operator char() const
            {
                if( m_reference )
                {
                    return (char)(*m_reference);
                }

                char retVal = false;

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToInteger( (void*)&m_baseValue );
                m_Mutex.Unlock();

                return retVal;
            }

            //
            virtual operator unsigned char() const
            {
                if( m_reference )
                {
                    return (unsigned char)(*m_reference);
                }

                unsigned char retVal = 0U;

                if( NULL != GetTypeInfo<retT>().ToIntegerCallback )
                {
                    m_Mutex.Lock();
                    retVal = GetTypeInfo<retT>().ToIntegerCallback( (void*)&m_baseValue );
                    m_Mutex.Unlock();
                }
                return retVal;
            }

            //
            virtual operator int() const
            {
                if( m_reference )
                {
                    return (int)(*m_reference);
                }

                int retVal = 0;

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToInteger( (void*)&m_baseValue );
                m_Mutex.Unlock();

                return retVal;
            }

            virtual operator unsigned int() const
            {
                if( m_reference )
                {
                    return (unsigned int)(*m_reference);
                }

                unsigned int retVal = 0U;

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToInteger( (void*)&m_baseValue );
                m_Mutex.Unlock();

                return retVal;
            }

            virtual operator float() const
            {
                if( m_reference )
                {
                    return (float)(*m_reference);
                }

                float retVal = 0.0F;

                if( NULL != GetTypeInfo<retT>().ToRealCallback )
                {
                    m_Mutex.Lock();
                    retVal = GetTypeInfo<retT>().ToRealCallback( (void*)&m_baseValue );
                    m_Mutex.Unlock();
                }
                return retVal;
            }

            virtual operator double() const
            {
                if( m_reference )
                {
                    return (double)(*m_reference);
                }

                double retVal = 0.0F;

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToReal( (void*)&m_baseValue );
                m_Mutex.Unlock();

                return retVal;
            }

            virtual operator std::string() const
            {
                if( m_reference )
                {
                    return (std::string)(*m_reference);
                }

                std::string retVal = "";

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToText( (void*)&m_baseValue );
                m_Mutex.Unlock();

                return retVal;
            }

            virtual std::string PreviousValueString() const
            {
                std::string retVal = "";

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToText( (void*)&m_baseValuePrew );
                m_Mutex.Unlock();

                return retVal;
            }

            virtual std::string CurentValueString() const
            {
                std::string retVal = "";

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToText( (void*)&m_baseValue );
                m_Mutex.Unlock();

                return retVal;
            }

            virtual int PreviousValueInteger() const
            {

            }

            virtual int CurentValueInteger() const
            {

            }

        private:
            retT m_baseValue;
            retT m_baseValuePrew;

            retT (classT::*GetValueCallback)();
            void (classT::*SetValueCallback)(retT);
    };
}

#endif
