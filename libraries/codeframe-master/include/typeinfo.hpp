#ifndef TYPEINFO_HPP_INCLUDED
#define TYPEINFO_HPP_INCLUDED

#include <string>

namespace codeframe
{
    enum eType
    {
        TYPE_NON = 0,
        TYPE_INT,
        TYPE_REAL,
        TYPE_TEXT
    };

    template<typename T>
    struct TypeInfo
    {
        TypeInfo( const char* typeName, const char* typeUser, const eType enumType ) :
            TypeCompName( typeName ),
            TypeUserName( typeUser ),
            TypeCode( enumType ),
            ToIntegerCallback( NULL ),
            ToTextCallback( NULL )
        {

        }

        void SetToIntegerCallback( int (*toIntegerCallback)(void* value) )
        {
            ToIntegerCallback = toIntegerCallback;
        }

        void SetToRealCallback( int (*toRealCallback)(void* value) )
        {
            ToRealCallback = toRealCallback;
        }

        const char* TypeCompName;
        const char* TypeUserName;
        const eType TypeCode;

        // Conversions to standard types
        int         ( *ToIntegerCallback )( void* value );
        float       ( *ToRealCallback    )( void* value );
        std::string ( *ToTextCallback    )( void* value );

        // Conversions from standard types
        void* ( *FromIntegerCallback )( int         value );
        void* ( *FromRealCallback    )( float       value );
        void* ( *FromTextCallback    )( std::string value );

        static const eType StringToTypeCode( std::string typeText );
    };

    template<typename T>
    TypeInfo<T> GetTypeInfo();

    #define REGISTER_TYPE(T,S) \
      template<> \
      TypeInfo<T> GetTypeInfo<T>() { TypeInfo<T> type(#T,S,TypeInfo<T>::StringToTypeCode(S)); return type; }

    void CODEFRAME_TYPES_INITIALIZE( void );
}

#endif // TYPEINFO_HPP_INCLUDED
