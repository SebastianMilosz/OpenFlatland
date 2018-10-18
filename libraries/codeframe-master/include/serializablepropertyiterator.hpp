#ifndef SERIALIZABLEPROPERTYITERATOR_HPP_INCLUDED
#define SERIALIZABLEPROPERTYITERATOR_HPP_INCLUDED

#include <iterator>

namespace codeframe
{
    class PropertyBase;
    class cPropertyManager;

    class PropertyIterator : public std::iterator<std::input_iterator_tag, PropertyBase*>
    {
        friend class cPropertyManager;

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
        PropertyIterator( cPropertyManager& pm, int n );

        cPropertyManager& m_PropertyManager;
        PropertyBase*     m_param;
        int               m_curId;
    };
}

#endif // SERIALIZABLEPROPERTYITERATOR_HPP_INCLUDED
