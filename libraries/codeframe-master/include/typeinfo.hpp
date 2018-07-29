#ifndef TYPEINFO_HPP_INCLUDED
#define TYPEINFO_HPP_INCLUDED

#include <string>
#include <MathUtilities.h>

namespace codeframe
{
    template <typename T, typename B>
    class TypeConversionFunctor
    {
        public:
            virtual T operator()( B& value ) const = 0;
    };

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

            void SetToStringCallback ( StringType  (*toStringCallback )( T value ) );
            void SetToIntegerCallback( IntegerType (*toIntegerCallback)( T value ) );
            void SetToRealCallback   ( RealType    (*toRealCallback   )( T value ) );

            T FromString ( StringType value  );
            T FromInteger( IntegerType value );
            T FromReal   ( RealType value    );

            StringType  ToString ( T value );
            IntegerType ToInteger( T value );
            RealType    ToReal   ( T value );

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

            T           ( *FromStringCallback )( StringType  value );
            StringType  ( *ToStringCallback   )( T           value );
            T           ( *FromIntegerCallback)( IntegerType value );
            IntegerType ( *ToIntegerCallback  )( T           value );
            T           ( *FromRealCallback   )( RealType    value );
            RealType    ( *ToRealCallback     )( T           value );
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
