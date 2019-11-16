#ifndef SERIALIZABLEPROPERTYITERATOR_HPP_INCLUDED
#define SERIALIZABLEPROPERTYITERATOR_HPP_INCLUDED

#include <iterator>

namespace codeframe
{
    class PropertyBase;
    class cPropertyList;

    class PropertyIterator : public std::iterator<std::input_iterator_tag, PropertyBase*>
    {
        friend class cPropertyList;

    public:
        PropertyIterator(const PropertyIterator& n);

        PropertyBase*     operator *();
        PropertyIterator& operator ++();

        // Operatory porownania
        bool operator< (const PropertyIterator& n);
        bool operator> (const PropertyIterator& n);
        bool operator<=(const PropertyIterator& n);
        bool operator>=(const PropertyIterator& n);
        bool operator==(const PropertyIterator& n);
        bool operator!=(const PropertyIterator& n);

    private:
        PropertyIterator( cPropertyList& pm, int n );

        cPropertyList& m_PropertyManager;
        PropertyBase*  m_param;
        int            m_curId;
    };
}

#endif // SERIALIZABLEPROPERTYITERATOR_HPP_INCLUDED
