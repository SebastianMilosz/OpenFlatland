#ifndef TYPEINFO_HPP_INCLUDED
#define TYPEINFO_HPP_INCLUDED

#include <string>

namespace codeframe
{
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
            TypeCode( enumType )
        {
            BytePrec = bytePrec;
            Sign = sign;
        }

        static void SetToIntegerCallback( int (*toIntegerCallback)(void* value, unsigned char bytePrec, bool sign) )
        {
            ToIntegerCallback = toIntegerCallback;
        }

        static void SetToRealCallback( int (*toRealCallback)(void* value, unsigned char bytePrec, bool sign) )
        {
            ToRealCallback = toRealCallback;
        }

        static void SetToTextCallback( int (*toTextCallback)(void* value, unsigned char bytePrec, bool sign) )
        {
            ToTextCallback = toTextCallback;
        }

        static int ToInteger( void* value )
        {
            if( NULL != ToIntegerCallback )
            {
                return ToIntegerCallback( value, BytePrec, Sign );
            }
        }

        static float ToReal( void* value )
        {
            if( NULL != ToRealCallback )
            {
                return ToRealCallback( value, BytePrec, Sign );
            }
        }

        static std::string ToText( void* value )
        {
            if( NULL != ToTextCallback )
            {
                return ToTextCallback( value, BytePrec, Sign );
            }
        }

        // Conversions from standard types
        static void* FromInteger( int value )
        {

            return NULL;
        }

        static void* FromReal( float value )
        {

            return NULL;
        }

        static void* FromText( std::string value )
        {

            return NULL;
        }

        const char* TypeCompName;
        const char* TypeUserName;
        const eType TypeCode;

        static const unsigned char BytePrec;
        static const bool Sign;

        // Conversions to standard types
        static int         ( *ToIntegerCallback )( void* value, unsigned char bytePrec, bool sign );
        static float       ( *ToRealCallback    )( void* value, unsigned char bytePrec, bool sign );
        static std::string ( *ToTextCallback    )( void* value, unsigned char bytePrec, bool sign );

        // Conversions from standard types
        static void* ( *FromIntegerCallback )( int         value );
        static void* ( *FromRealCallback    )( float       value );
        static void* ( *FromTextCallback    )( std::string value );

        static const eType StringToTypeCode( std::string typeText );
    };

    template <typename T>
    TypeInfo<T>::ToIntegerCallback = NULL;

    template <typename T>
    TypeInfo<T>::ToRealCallback = NULL;

    template <typename T>
    TypeInfo<T>::ToTextCallback = NULL;

    template <typename T>
    TypeInfo<T>::FromIntegerCallback = NULL;

    template <typename T>
    TypeInfo<T>::FromRealCallback = NULL;

    template <typename T>
    TypeInfo<T>::FromTextCallback = NULL;

    template<typename T>
    TypeInfo<T> GetTypeInfo();

    #define REGISTER_TYPE(T,S,PrecByte,Sign) \
      template<> \
      TypeInfo<T> GetTypeInfo<T>() { TypeInfo<T> type(#T,S,TypeInfo<T>::StringToTypeCode(S),PrecByte,Sign); return type; }

    void CODEFRAME_TYPES_INITIALIZE( void );
}

#endif // TYPEINFO_HPP_INCLUDED
