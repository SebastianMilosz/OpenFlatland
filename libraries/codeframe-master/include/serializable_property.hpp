#ifndef _SERIALIZABLE_PROPERTY_H
#define _SERIALIZABLE_PROPERTY_H

#include "serializable_property_base.hpp"
#include "serializable_property_selection.hpp"
#include "extpoint2d.hpp"
#include "extvector.hpp"

#include <LoggerUtilities.h>
#include <functional>
#include <climits>
#include <string>
#include <typeinfo>     // operator typeid

namespace codeframe
{
    class Object;

    template <typename PROPERTY_TYPE>
    struct PropertyIndex
    {
        friend class Property<PROPERTY_TYPE>;

        public:
            PropertyIndex& operator=( std::size_t val )
            {
                m_index = val;
                return *this;
            }

            operator std::size_t() const
            {
                return m_index;
            }

        private:
            PropertyIndex(Property<PROPERTY_TYPE>& prop) :
                m_index(0U),
                m_prop(prop)
                {

                }
            PropertyIndex(Property<PROPERTY_TYPE>& prop, std::size_t index) :
                m_index(index),
                m_prop(prop)
                {
                }

            std::size_t m_index = 0U;
            Property<PROPERTY_TYPE>& m_prop;
    };

    /*****************************************************************************
     * @class Property
     *****************************************************************************/
    template <typename PROPERTY_TYPE>
    class Property : public PropertyBase
    {
        public:
            Property( Object* parentpc,
                      const std::string& name,
                      PROPERTY_TYPE val,
                      cPropertyInfo info,
                      std::function<const PROPERTY_TYPE&()> getConstValueFunc = nullptr,
                      std::function<void(PROPERTY_TYPE)> setValueFunc = nullptr,
                      std::function<PROPERTY_TYPE&()> getValueFunc = nullptr
                     ) : PropertyBase( parentpc, name, GetTypeInfo<PROPERTY_TYPE>().GetTypeCode(), info ),
                     m_baseValue(val),
                     m_Index(*this, 0U),
                     m_GetConstValueFunction(getConstValueFunc),
                     m_GetValueFunction(getValueFunc),
                     m_SetValueFunction(setValueFunc)
            {
            }

            // Destructor
            virtual ~Property()
            {

            }

            PropertyIndex<PROPERTY_TYPE>& Index()
            {
                return m_Index;
            }

            const PROPERTY_TYPE& GetConstValue() const
            {
                if ( m_reference )
                {
                    auto referencePropertySelection = dynamic_cast<codeframe::PropertySelection* >(smart_ptr_getRaw(m_reference));
                    if (referencePropertySelection)
                    {
                        return referencePropertySelection->GetConstValue<PROPERTY_TYPE>();
                    }
                }

                if ( static_cast<bool>(m_GetConstValueFunction) )
                {
                    return m_GetConstValueFunction();
                }
                else if ( m_GetValueFunction )
                {
                    return m_GetValueFunction();
                }
                return m_baseValue;
            }

            PROPERTY_TYPE& GetValue()
            {
                if ( m_reference )
                {
                    auto referencePropertySelection = dynamic_cast<codeframe::PropertySelection* >(smart_ptr_getRaw(m_reference));
                    if (referencePropertySelection)
                    {
                        return referencePropertySelection->GetValue<PROPERTY_TYPE>();
                    }
                }

                if ( static_cast<bool>(m_GetValueFunction) )
                {
                    return m_GetValueFunction();
                }
                return m_baseValue;
            }

            constexpr bool_t IsValueReadOnly() const
            {
                if ( static_cast<bool>(m_GetValueFunction) == false )
                {
                    if ( static_cast<bool>(m_GetConstValueFunction) )
                    {
                        return true;
                    }
                }
                return false;
            }

            PROPERTY_TYPE& GetBaseValue()
            {
                return m_baseValue;
            }

            // Copy operator
            Property( const Property& sval ) : PropertyBase( sval ),
                m_Index(*this, sval.m_Index.m_index)
            {
                // Values
                m_baseValue = sval.GetConstValue();
                m_baseValuePrew = sval.m_baseValuePrew;

                // Funktors
                m_GetConstValueFunction = sval.m_GetConstValueFunction;
                m_GetValueFunction      = sval.m_GetValueFunction;
                m_SetValueFunction      = sval.m_SetValueFunction;
            }

            // Comparison operators
            bool_t operator==(const Property& sval) const
            {
                m_Mutex.Lock();
                bool_t retVal = false;
                if ( GetConstValue() == sval.GetConstValue() )
                {
                    retVal = true;
                }
                m_Mutex.Unlock();

                return retVal;
            }

            // Comparison operators
            bool_t operator!=(const Property& sval) const
            {
                m_Mutex.Lock();
                bool_t retVal = false;
                if ( GetConstValue() != sval.GetConstValue() )
                {
                    retVal = true;
                }
                m_Mutex.Unlock();

                return retVal;
            }

            // Comparison operators
            bool_t operator==(const int& sval) const override
            {
                bool_t retVal = false;

                m_Mutex.Lock();

                if( GetTypeInfo<PROPERTY_TYPE>().ToInteger( GetConstValue() ) == sval )
                {
                    retVal = true;
                }

                m_Mutex.Unlock();

                return retVal;
            }

            bool_t operator!=(const int& sval) const override
            {
                return !(*this==sval);
            }

            // Copy operator
            virtual Property& operator=( const Property& val )
            {
                this->PropertyBase::operator=( val );

                // Funktors
                m_GetConstValueFunction = val.m_GetConstValueFunction;
                m_GetValueFunction      = val.m_GetValueFunction;
                m_SetValueFunction      = val.m_SetValueFunction;

                m_baseValue = val.m_baseValue;
                m_baseValuePrew = val.m_baseValuePrew;

                // Values
                if ( IsChanged() == true  )
                {
                    EmitChanges();
                }

                return *this;
            }

            // From fundamental type bool
            Property& operator=( bool_t val ) override
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetConstValue();
                    m_baseValue = GetTypeInfo<PROPERTY_TYPE>().FromInteger( val );
                    m_Mutex.Unlock();

                    // Values external
                    if ( IsChanged() == true  )
                    {
                        EmitChanges();
                    }

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From fundamental type char
            Property& operator=( char val ) override
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetConstValue();
                    m_baseValue = GetTypeInfo<PROPERTY_TYPE>().FromInteger( val );
                    m_Mutex.Unlock();

                    // Values external
                    if ( IsChanged() == true  )
                    {
                        EmitChanges();
                    }

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From fundamental type unsigned char
            Property& operator=(unsigned char val) override
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetConstValue();
                    m_baseValue = GetTypeInfo<PROPERTY_TYPE>().FromInteger( val );
                    m_Mutex.Unlock();

                    // Values external
                    if ( IsChanged() == true  )
                    {
                        EmitChanges();
                    }

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From fundamental type int
            Property& operator=( int val ) override
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetConstValue();
                    m_baseValue = GetTypeInfo<PROPERTY_TYPE>().FromInteger( val );
                    m_Mutex.Unlock();

                    // Values external
                    if ( IsChanged() == true  )
                    {
                        EmitChanges();
                    }

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From fundamental type unsigned int
            Property& operator=( unsigned int val ) override
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetConstValue();
                    m_baseValue = GetTypeInfo<PROPERTY_TYPE>().FromInteger( val );
                    m_Mutex.Unlock();

                    // Values external
                    if ( IsChanged() == true  )
                    {
                        EmitChanges();
                    }

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From fundamental type float
            Property& operator=(float val) override
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetConstValue();
                    m_baseValue = GetTypeInfo<PROPERTY_TYPE>().FromReal( val );
                    m_Mutex.Unlock();

                    // Values external
                    if ( IsChanged() == true  )
                    {
                        EmitChanges();
                    }

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From fundamental type double
            Property& operator=(double val) override
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetConstValue();
                    m_baseValue = GetTypeInfo<PROPERTY_TYPE>().FromReal( val );
                    m_Mutex.Unlock();

                    // Values external
                    if ( IsChanged() == true  )
                    {
                        EmitChanges();
                    }

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // From extended type std::string
            Property& operator=( const std::string& val ) override
            {
                if ( Info().GetEnable() == true )
                {
                    m_Mutex.Lock();
                    m_baseValuePrew = GetConstValue();
                    m_baseValue = GetTypeInfo<PROPERTY_TYPE>().FromString( val );
                    m_Mutex.Unlock();

                    // Values external
                    if ( IsChanged() == true  )
                    {
                        EmitChanges();
                    }

                    // Przypisanie wartosci zdalnej referencji
                    if ( m_reference )
                    {
                        *m_reference = val;
                    }
                }
                return *this;
            }

            // ++
            Property& operator++() override
            {
                (*this) = (int)(*this) + 1U;

                // actual increment takes place here
                return *this;
            }

            // --
            Property& operator--() override
            {
                (*this) = (int)(*this) - 1U;

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
                m_baseValuePrew = GetConstValue();
                m_baseValue = GetTypeInfo<PROPERTY_TYPE>().AddOperator( GetConstValue(), rhs.GetConstValue() );
                m_Mutex.Unlock();

                // Values external
                if ( IsChanged() == true  )
                {
                    EmitChanges();
                }

                return *this;
            }

            // -
            virtual Property  operator-(const Property& rhs)
            {
                m_Mutex.Lock();
                IntegerType valA = GetTypeInfo<PROPERTY_TYPE>().ToInteger( GetConstValue()     );
                IntegerType valB = GetTypeInfo<PROPERTY_TYPE>().ToInteger( rhs.GetConstValue() );

                PROPERTY_TYPE valueT = GetTypeInfo<PROPERTY_TYPE>().FromInteger( valA - valB );

                m_baseValuePrew = GetConstValue();
                m_baseValue = valueT;

                m_Mutex.Unlock();

                // Values external
                if ( IsChanged() == true  )
                {
                    EmitChanges();
                }

                return *this;
            }

            // +=
            virtual Property& operator+=(const int rhs)
            {
                Property<PROPERTY_TYPE> prop(*this);
                prop = rhs;

                *this = *this + prop;
                return *this;
            }

            // -=
            virtual Property& operator-=(const int rhs)
            {
                Property<PROPERTY_TYPE> prop(*this);
                prop = rhs;

                *this = *this - prop;
                return *this;
            }

            //
            virtual Property  operator++(int)
            {
                Property<PROPERTY_TYPE> prop(*this); // copy
                operator++();                       // pre-increment
                return prop;                        // return old value
            }

            //
            virtual Property  operator--(int)
            {
                Property<PROPERTY_TYPE> prop(*this); // copy
                operator--();                       // pre-decrement
                return prop;                        // return old value
            }

            // Fundamental types casting operators
            operator bool() const override
            {
                if( m_reference )
                {
                    return (bool)(*m_reference);
                }

                bool_t retVal = false;

                m_Mutex.Lock();
                retVal = GetTypeInfo<PROPERTY_TYPE>().ToInteger( GetConstValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            operator char() const override
            {
                if( m_reference )
                {
                    return (char)(*m_reference);
                }

                char retVal = false;

                m_Mutex.Lock();
                retVal = GetTypeInfo<PROPERTY_TYPE>().ToInteger( GetConstValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            //
            operator unsigned char() const override
            {
                if( m_reference )
                {
                    return (unsigned char)(*m_reference);
                }

                unsigned char retVal = 0U;

                m_Mutex.Lock();
                retVal = GetTypeInfo<PROPERTY_TYPE>().ToInteger( GetConstValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            //
            operator int() const override
            {
                if( m_reference )
                {
                    return (int)(*m_reference);
                }

                int retVal(0);

                m_Mutex.Lock();
                retVal = GetTypeInfo<PROPERTY_TYPE>().ToInteger( GetConstValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            operator unsigned int() const override
            {
                if( m_reference )
                {
                    return (unsigned int)(*m_reference);
                }

                unsigned int retVal(0U);

                m_Mutex.Lock();
                retVal = GetTypeInfo<PROPERTY_TYPE>().ToInteger( GetConstValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            operator float() const override
            {
                if( m_reference )
                {
                    return (float)(*m_reference);
                }

                float retVal(0.0F);

                m_Mutex.Lock();
                retVal = GetTypeInfo<PROPERTY_TYPE>().ToReal( GetConstValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            operator double() const override
            {
                if( m_reference )
                {
                    return (double)(*m_reference);
                }

                double retVal(0.0F);

                m_Mutex.Lock();
                retVal = GetTypeInfo<PROPERTY_TYPE>().ToReal( GetConstValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            operator std::string() const override
            {
                if( m_reference )
                {
                    return (std::string)(*m_reference);
                }

                std::string retVal("");

                m_Mutex.Lock();
                retVal = GetTypeInfo<PROPERTY_TYPE>().ToString( GetConstValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            std::string PreviousValueString() const override
            {
                std::string retVal("");

                m_Mutex.Lock();
                retVal = GetTypeInfo<PROPERTY_TYPE>().ToString( m_baseValuePrew );
                m_Mutex.Unlock();

                return retVal;
            }

            std::string CurentValueString() const override
            {
                std::string retVal("");

                m_Mutex.Lock();
                retVal = GetTypeInfo<PROPERTY_TYPE>().ToString( GetConstValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            int PreviousValueInteger() const override
            {
                int retVal(0);

                m_Mutex.Lock();
                retVal = GetTypeInfo<PROPERTY_TYPE>().ToInteger( m_baseValuePrew );
                m_Mutex.Unlock();

                return retVal;
            }

            int CurentValueInteger() const override
            {
                int retVal(0);

                m_Mutex.Lock();
                retVal = GetTypeInfo<PROPERTY_TYPE>().ToInteger( GetConstValue() );
                m_Mutex.Unlock();

                return retVal;
            }

            std::string TypeString() const
            {
                return std::string( GetTypeInfo<PROPERTY_TYPE>().GetTypeUserName() );
            }

            void CommitChanges()
            {
                PropertyBase::CommitChanges();

                m_Mutex.Lock();
                m_baseValuePrew = GetConstValue();
                m_Mutex.Unlock();
            }

            bool_t IsChanged() const override
            {
                if ( (PropertyBase::IsChanged()) || (m_baseValuePrew != GetConstValue()) )
                {
                    return true;
                }

                return false;
            }

        protected:
            void EmitChanges() override
            {
                if ( m_SetValueFunction )
                {
                    m_SetValueFunction( m_baseValue );
                }

                if ( m_propertyInfo.IsEventEnable() )
                {
                    signalChanged.Emit( this );
                }
            }

        private:
            PROPERTY_TYPE m_baseValue;
            PROPERTY_TYPE m_baseValuePrew;

            PropertyIndex<PROPERTY_TYPE> m_Index;

            std::function<const PROPERTY_TYPE&()> m_GetConstValueFunction;
            std::function<PROPERTY_TYPE&()>       m_GetValueFunction;
            std::function<void(PROPERTY_TYPE)>    m_SetValueFunction;
    };
}

#endif  // _SERIALIZABLE_PROPERTY_H
