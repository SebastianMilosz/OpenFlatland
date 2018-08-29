#ifndef TYPEINFO_HPP_INCLUDED
#define TYPEINFO_HPP_INCLUDED

#include <string>
#include <MathUtilities.h>

#include "extpoint2d.hpp"
#include "extvector.hpp"

namespace codeframe
{
    template <typename T, typename B>
    class TypeConversionFunctor
    {
        public:
            virtual T operator()( B& value ) const = 0;
    };

    enum eType
    {
        TYPE_NON = 0,   ///< No type
        TYPE_INT,       ///< Fundamental integer type
        TYPE_REAL,      ///< Fundamental real type
        TYPE_TEXT,      ///< Fundamental text type
        TYPE_IVECTOR    ///< Vector of integer values
    };

    /// @todo Przeprojektowac!!!!
    struct VariantValue
    {
        bool IsName( std::string name ) const
        {
            if (name == Name)
            {
                return true;
            }
            return false;
        }

        const eType GetType() const
        {
            return Type;
        }

        IntegerType IntegerValue() const
        {
            return Value.Integer;
        }

        eType       Type;   ///< string to type conversion
        std::string Name;   ///< variable name

        union ValueUnion
        {
            int     Integer;
            double  Real;
        } Value;

        std::string ValueString;  ///< variable value
    };

    template<typename T>
    class TypeInfo
    {
        public:
            TypeInfo( const char* typeName, const char* typeUser, const eType enumType );

            static const eType StringToTypeCode( std::string typeText );

            void SetFromStringCallback ( T (*fromStringCallback )( StringType  value ) );
            void SetFromIntegerCallback( T (*fromIntegerCallback)( IntegerType value ) );
            void SetFromRealCallback   ( T (*fromRealCallback   )( RealType    value ) );

            void SetToStringCallback ( StringType  (*toStringCallback )( const T& value ) );
            void SetToIntegerCallback( IntegerType (*toIntegerCallback)( const T& value ) );
            void SetToRealCallback   ( RealType    (*toRealCallback   )( const T& value ) );

            T FromString( StringType value )
            {
                if ( NULL != FromStringCallback )
                {
                    return FromStringCallback( value );
                }
                return T();
            }

            T FromInteger( IntegerType value )
            {
                if ( NULL != FromIntegerCallback )
                {
                    return FromIntegerCallback( value );
                }
                return T();
            }

            T FromReal( RealType value )
            {
                if ( NULL != FromRealCallback )
                {
                    return FromRealCallback( value );
                }
                return T();
            }

            StringType ToString( T value )
            {
                if ( NULL != ToStringCallback )
                {
                    return ToStringCallback( value );
                }
                return StringType("");
            }

            IntegerType ToInteger( T value )
            {
                if ( NULL != ToIntegerCallback )
                {
                    return ToIntegerCallback( value );
                }
                return IntegerType();
            }

            RealType ToReal( T value )
            {
                if ( NULL != ToRealCallback )
                {
                    return ToRealCallback( value );
                }
                return 0.0F;
            }

            const eType GetTypeCode() const
            {
                return TypeCode;
            }

            const char* GetTypeUserName() const
            {
                return TypeUserName;
            }

        private:
            const char* TypeCompName;
            const char* TypeUserName;
            const eType TypeCode;

            T          ( *FromStringCallback )( StringType  value );
            StringType ( *ToStringCallback   )( const T&    value );
            T          ( *FromIntegerCallback)( IntegerType value );
            IntegerType( *ToIntegerCallback  )( const T&    value );
            T          ( *FromRealCallback   )( RealType    value );
            RealType   ( *ToRealCallback     )( const T&    value );
    };

    class TypeInitializer
    {
        public:
            TypeInitializer();
    };

    template<typename T>
    TypeInfo<T>& GetTypeInfo();

    #define REGISTER_TYPE(T,S) \
      template<> \
      TypeInfo<T>& GetTypeInfo<T>() \
      { \
          static TypeInfo<T> type(#T,S,TypeInfo<T>::StringToTypeCode(S)); \
          return type; \
      }

    void CODEFRAME_TYPES_INITIALIZE( void );
}

#endif // TYPEINFO_HPP_INCLUDED
