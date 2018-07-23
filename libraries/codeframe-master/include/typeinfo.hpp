#ifndef TYPEINFO_HPP_INCLUDED
#define TYPEINFO_HPP_INCLUDED

#include <string>

namespace codeframe
{
    typedef int IntegerType;
    typedef double RealType;
    typedef std::string StringType;

    enum eType
    {
        TYPE_NON = 0,   ///< No type
        TYPE_INT,       ///< Fundamental integer type
        TYPE_REAL,      ///< Fundamental real type
        TYPE_TEXT,      ///< Fundamental text type
        TYPE_EXTENDED   ///< Extended type inherit from ExtendTypeInterface
    };

    template<typename T>
    struct TypeInfo
    {
        TypeInfo( const char* typeName, const char* typeUser, const eType enumType, unsigned char bytePrec = 4, bool sign = true ) :
            TypeCompName( typeName ),
            TypeUserName( typeUser ),
            TypeCode( enumType ),
            BytePrec( bytePrec ),
            Sign( sign ),
            FromStringCallback( NULL ),
            ToStringCallback( NULL )
        {
        }

        void SetFromStringCallback( T (*fromStringCallback)( StringType value ) )
        {
            FromStringCallback = fromStringCallback;
        }

        T FromString( StringType value )
        {
            if ( NULL != FromStringCallback )
            {
                return FromStringCallback( value );
            }
            return T();
        }

        void SetToStringCallback( StringType (*toStringCallback)( T value ) )
        {
            ToStringCallback = toStringCallback;
        }

        StringType ToString( T value )
        {
            if ( NULL != ToStringCallback )
            {
                return ToStringCallback( value );
            }
            return StringType("");
        }

        void SetFromIntegerCallback( T (*fromIntegerCallback)( int value ) )
        {
            FromIntegerCallback = fromIntegerCallback;
        }

        T FromInteger( int value )
        {
            if ( NULL != FromIntegerCallback )
            {
                return FromIntegerCallback( value );
            }
            return T();
        }

        void SetToIntegerCallback( int (*toIntegerCallback)( T value ) )
        {
            ToIntegerCallback = toIntegerCallback;
        }

        int ToInteger( T value )
        {
            if ( NULL != ToIntegerCallback )
            {
                return ToIntegerCallback( value );
            }
            return 0;
        }

        void SetFromRealCallback( T (*fromRealCallback)( double value ) )
        {
            FromRealCallback = fromRealCallback;
        }

        T FromReal( double value )
        {
            if ( NULL != FromRealCallback )
            {
                return FromRealCallback( value );
            }
            return T();
        }

        void SetToRealCallback( double (*toRealCallback)( T value ) )
        {
            ToRealCallback = toRealCallback;
        }

        double ToReal( T value )
        {
            if ( NULL != ToRealCallback )
            {
                return ToRealCallback( value );
            }
            return 0.0F;
        }

        const char* TypeCompName;
        const char* TypeUserName;
        const eType TypeCode;

        unsigned char BytePrec;
        bool Sign;

        T           ( *FromStringCallback )( StringType  value );
        StringType  ( *ToStringCallback   )( T           value );
        T           ( *FromIntegerCallback)( int         value );
        int         ( *ToIntegerCallback  )( T           value );
        T           ( *FromRealCallback   )( double      value );
        double      ( *ToRealCallback     )( T           value );

        static const eType StringToTypeCode( std::string typeText );
    };

    template<typename T>
    TypeInfo<T>& GetTypeInfo();

    #define REGISTER_TYPE(T,S,PrecByte,Sign) \
      template<> \
      TypeInfo<T>& GetTypeInfo<T>() \
      { \
          static TypeInfo<T> type(#T,S,TypeInfo<T>::StringToTypeCode(S),PrecByte,Sign); \
          return type; \
      }

    void CODEFRAME_TYPES_INITIALIZE( void );
}

#endif // TYPEINFO_HPP_INCLUDED
