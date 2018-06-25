#ifndef TYPEINFO_HPP_INCLUDED
#define TYPEINFO_HPP_INCLUDED

#include <string>

namespace codeframe
{
    enum eType
    {
        TYPE_NON = 0,
        TYPE_CHAR,
        TYPE_INT,
        TYPE_REAL,
        TYPE_TEXT,
        TYPE_IMAGE
    };

    template<typename T>
    struct TypeInfo
    {
        const char* TypeCompName;
        const char* TypeUserName;
        const eType TypeCode;

        static const eType StringToTypeCode( std::string typeText );
    };

    template<typename T>
    const TypeInfo<T> GetTypeInfo();

    #define REGISTER_TYPE(T,S) \
      template<> \
      const TypeInfo<T> GetTypeInfo<T>() { TypeInfo<T> type = {#T,S,TypeInfo<T>::StringToTypeCode(S)}; return type; }
}

#endif // TYPEINFO_HPP_INCLUDED
