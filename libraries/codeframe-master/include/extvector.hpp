#ifndef SERIALIZABLEPROPERTYTABLE_HPP_INCLUDED
#define SERIALIZABLEPROPERTYTABLE_HPP_INCLUDED

#include "typeinterface.hpp"

namespace codeframe
{
    template<class T>
    class Vector : public TypeInterface
    {
        public:
                     Vector();
                     Vector( unsigned int cnt );
                     Vector( const Vector& other );
            virtual ~Vector();

            virtual void FromStringCallback ( StringType  value );
            virtual void FromIntegerCallback( IntegerType value );
            virtual void FromRealCallback( RealType value );

            virtual StringType ToStringCallback() const;
            virtual IntegerType ToIntegerCallback() const;
            virtual RealType ToRealCallback() const;

            virtual Vector& operator=(const Vector& other);
            virtual Vector& operator+(const Vector& rhs);
            virtual bool    operator==(const Vector& sval);
            virtual bool    operator!=(const Vector& sval);

        private:
            T* m_table;
    };
}

#endif // SERIALIZABLEPROPERTYTABLE_HPP_INCLUDED
