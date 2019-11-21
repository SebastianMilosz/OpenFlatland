#include "serializable_property_iterator.hpp"

#include "serializable_property_list.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief Konstruktor kopiujacy
     **
    ******************************************************************************/
    PropertyIterator::PropertyIterator(const PropertyIterator& n) :
        m_PropertyManager(n.m_PropertyManager),
        m_param(n.m_param),
        m_curId(n.m_curId)
    {

    }

    /*****************************************************************************/
    /**
      * @brief Operator wskaznikowy wyodrebnienia wskazywanej wartosci
     **
    ******************************************************************************/
    PropertyBase* PropertyIterator::operator *()
    {
        m_param = m_PropertyManager.GetObjectFieldValue( m_curId );
        return m_param;
    }

    /*****************************************************************************/
    /**
      * @brief Operator inkrementacji (przejscia na kolejne pole)
     **
    ******************************************************************************/
    PropertyIterator& PropertyIterator::operator ++()
    {
        if ( m_curId < m_PropertyManager.GetObjectFieldCnt() )
        {
            ++m_curId;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyIterator::operator< (const PropertyIterator& n)
    {
        return n.m_curId <  m_curId;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyIterator::operator> (const PropertyIterator& n)
    {
        return n.m_curId >  m_curId;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyIterator::operator<=(const PropertyIterator& n)
    {
        return !(n.m_curId >  m_curId);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyIterator::operator>=(const PropertyIterator& n)
    {
        return !(n.m_curId <  m_curId);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyIterator::operator==(const PropertyIterator& n)
    {
        return n.m_curId == m_curId;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyIterator::operator!=(const PropertyIterator& n)
    {
        return !(n.m_curId == m_curId);
    }

    /*****************************************************************************/
    /**
      * @brief Konstruktor bazowy prywatny bo tylko cSerializable moze tworzyc swoje iteratory
     **
    ******************************************************************************/
    PropertyIterator::PropertyIterator( cPropertyList& pm, int n ) :
        m_PropertyManager( pm ),
        m_curId( n )
    {

    }
}
