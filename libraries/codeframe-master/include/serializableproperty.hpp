#ifndef _SERIALIZABLEPROPERTY_H
#define _SERIALIZABLEPROPERTY_H

#include "serializablepropertybase.hpp"
#include "extpoint2d.hpp"
#include "extvector.hpp"

#include <climits>
#include <string>

namespace codeframe
{
    class cSerializable;

    /*****************************************************************************
     * @class Property
     *****************************************************************************/
    template <typename retT, typename classT = cSerializable >
    class Property : public PropertyBase
    {
        public:
            Property( cSerializable* parentpc,
                      const std::string& name,
                      retT val,
                      cPropertyInfo info,
                      classT* contextObject = NULL,
                      const retT& (classT::*getValue)() = NULL,
                      void        (classT::*setValue)(retT) = NULL
                     ) : PropertyBase( parentpc, name, GetTypeInfo<retT>().GetTypeCode(), info )
            {
                ContextObject    = contextObject;
                GetValueCallback = getValue;
                SetValueCallback = setValue;
                m_baseValue      = val;
            }

            // Destructor
            virtual ~Property()
            {

            }

            retT GetValue() const
            {
                if ( (NULL != GetValueCallback) && (NULL != ContextObject) )
                {
                    return (ContextObject->*GetValueCallback)();
                }
                return m_baseValue;
            }

            void SetValue( const retT& value )
            {
                if ( (NULL != SetValueCallback) && (NULL != ContextObject) )
                {
                    (ContextObject->*SetValueCallback)( value );
                }

                if ( m_propertyInfo.IsEventEnable() )
                {
                    signalChanged.Emit( this );
                }
            }

            // Copy operator
            Property( const Property& sval ) : PropertyBase( sval )
            {
                // Values
                m_baseValue = sval.GetValue();
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
                if ( GetValue() == sval.GetValue() )
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
                if ( GetValue() != sval.GetValue() )
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

                if( GetTypeInfo<retT>().ToInteger( GetValue() ) == sval )
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
            virtual Property& operator=( const Property& val )
            {
                this->PropertyBase::operator=( val );

                // Values
                SetValue( val.GetValue() );

                // Funktors
                GetValueCallback = val.GetValueCallback;
                SetValueCallback = val.SetValueCallback;

                return *this;
            }

            // From fundamental type bool
            virtual Property& operator=( bool val )
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetValue();
                    m_baseValue = GetTypeInfo<retT>().FromInteger( val );

                    // Values external
                    SetValue( m_baseValue );

                    m_Mutex.Unlock();

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From fundamental type char
            virtual Property& operator=( char val )
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetValue();
                    m_baseValue = GetTypeInfo<retT>().FromInteger( val );

                    // Values external
                    SetValue( m_baseValue );

                    m_Mutex.Unlock();

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From fundamental type unsigned char
            virtual Property& operator=(unsigned char val)
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetValue();
                    m_baseValue = GetTypeInfo<retT>().FromInteger( val );

                    // Values external
                    SetValue( m_baseValue );

                    m_Mutex.Unlock();

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From fundamental type int
            virtual Property& operator=( int val )
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetValue();
                    m_baseValue = GetTypeInfo<retT>().FromInteger( val );

                    // Values external
                    SetValue( m_baseValue );

                    m_Mutex.Unlock();

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From fundamental type unsigned int
            virtual Property& operator=( unsigned int val )
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetValue();
                    m_baseValue = GetTypeInfo<retT>().FromInteger( val );

                    // Values external
                    SetValue( m_baseValue );

                    m_Mutex.Unlock();

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From fundamental type float
            virtual Property& operator=(float val)
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetValue();
                    m_baseValue = GetTypeInfo<retT>().FromReal( val );

                    // Values external
                    SetValue( m_baseValue );

                    m_Mutex.Unlock();

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From fundamental type double
            virtual Property& operator=(double val)
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetValue();
                    m_baseValue = GetTypeInfo<retT>().FromReal( val );

                    // Values external
                    SetValue( m_baseValue );

                    m_Mutex.Unlock();

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From extended type std::string
            virtual Property& operator=( std::string& val )
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetValue();
                    m_baseValue = GetTypeInfo<retT>().FromString( val );

                    // Values external
                    SetValue( m_baseValue );

                    m_Mutex.Unlock();

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
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
                m_baseValuePrew = GetValue();
                m_baseValue = GetTypeInfo<retT>().AddOperator( GetValue(), rhs.GetValue() );
                m_Mutex.Unlock();

                // Values external
                SetValue( m_baseValue );

                return *this;
            }

            // -
            virtual Property  operator-(const Property& rhs)
            {
                m_Mutex.Lock();
                IntegerType valA = GetTypeInfo<retT>().ToInteger( GetValue()     );
                IntegerType valB = GetTypeInfo<retT>().ToInteger( rhs.GetValue() );

                retT valueT = GetTypeInfo<retT>().FromInteger( valA - valB );

                m_baseValuePrew = GetValue();
                m_baseValue = valueT;

                m_Mutex.Unlock();

                // Values external
                SetValue( m_baseValue );

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
                retVal = GetTypeInfo<retT>().ToInteger( GetValue() );
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
                retVal = GetTypeInfo<retT>().ToInteger( GetValue() );
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

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToInteger( GetValue() );
                m_Mutex.Unlock();

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
                retVal = GetTypeInfo<retT>().ToInteger( GetValue() );
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
                retVal = GetTypeInfo<retT>().ToInteger( GetValue() );
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

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToReal( GetValue() );
                m_Mutex.Unlock();

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
                retVal = GetTypeInfo<retT>().ToReal( GetValue() );
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
                retVal = GetTypeInfo<retT>().ToString( GetValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            virtual std::string PreviousValueString() const
            {
                std::string retVal = "";

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToString( m_baseValuePrew );
                m_Mutex.Unlock();

                return retVal;
            }

            virtual std::string CurentValueString() const
            {
                std::string retVal = "";

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToString( GetValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            virtual int PreviousValueInteger() const
            {
                int retVal = 0;

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToInteger( m_baseValuePrew );
                m_Mutex.Unlock();

                return retVal;
            }

            virtual int CurentValueInteger() const
            {
                int retVal = 0;

                m_Mutex.Lock();
                retVal = GetTypeInfo<retT>().ToInteger( GetValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            std::string TypeString() const
            {
                return std::string( GetTypeInfo<retT>().GetTypeUserName() );
            }

            void CommitChanges()
            {
                PropertyBase::CommitChanges();

                m_Mutex.Lock();
                m_baseValuePrew = GetValue();
                m_Mutex.Unlock();
            }

            bool IsChanged() const
            {
                if ( m_baseValuePrew != GetValue() )
                {
                    return true;
                }

                return false;
            }



        private:
            retT m_baseValue;
            retT m_baseValuePrew;

            classT* ContextObject;
            const retT& (classT::*GetValueCallback)();
            void        (classT::*SetValueCallback)(retT);
    };
}

#endif
