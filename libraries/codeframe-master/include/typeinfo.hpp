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
            TypeCode( enumType ),
            ToIntegerCallback( NULL ),
            ToRealCallback( NULL ),
            ToTextCallback( NULL ),
            FromIntegerCallback( NULL ),
            FromRealCallback( NULL ),
            FromTextCallback( NULL ),
            BytePrec( bytePrec ),
            Sign( sign )
        {

        }

        void SetToIntegerCallback( int (*toIntegerCallback)(void* value, unsigned char bytePrec, bool sign) )
        {
            ToIntegerCallback = toIntegerCallback;
        }

        void SetToRealCallback( int (*toRealCallback)(void* value, unsigned char bytePrec, bool sign) )
        {
            ToRealCallback = toRealCallback;
        }

        void SetToTextCallback( int (*toTextCallback)(void* value, unsigned char bytePrec, bool sign) )
        {
            ToTextCallback = toTextCallback;
        }

        int ToInteger( void* value )
        {
            if( NULL != ToIntegerCallback )
            {
                return ToIntegerCallback( value, BytePrec, Sign );
            }
        }

        float ToReal( void* value )
        {
            if( NULL != ToRealCallback )
            {
                return ToRealCallback( value, BytePrec, Sign );
            }
        }

        std::string ToText( void* value )
        {
            if( NULL != ToTextCallback )
            {
                return ToTextCallback( value, BytePrec, Sign );
            }
        }

        // Conversions from standard types
        void* FromInteger( int value )
        {

            return NULL;
        }

        void* FromReal( float value )
        {

            return NULL;
        }

        void* FromText( std::string value )
        {

            return NULL;
        }

        const char* TypeCompName;
        const char* TypeUserName;
        const eType TypeCode;

        const unsigned char BytePrec;
        const bool Sign;

        // Conversions to standard types
        int         ( *ToIntegerCallback )( void* value, unsigned char bytePrec, bool sign );
        float       ( *ToRealCallback    )( void* value, unsigned char bytePrec, bool sign );
        std::string ( *ToTextCallback    )( void* value, unsigned char bytePrec, bool sign );

        // Conversions from standard types
        void* ( *FromIntegerCallback )( int         value );
        void* ( *FromRealCallback    )( float       value );
        void* ( *FromTextCallback    )( std::string value );

        static const eType StringToTypeCode( std::string typeText );
    };

    template<typename T>
    TypeInfo<T> GetTypeInfo();

    #define REGISTER_TYPE(T,S,PrecByte,Sign) \
      template<> \
      TypeInfo<T> GetTypeInfo<T>() { TypeInfo<T> type(#T,S,TypeInfo<T>::StringToTypeCode(S),PrecByte,Sign); return type; }

    void CODEFRAME_TYPES_INITIALIZE( void );
}

#endif // TYPEINFO_HPP_INCLUDED
