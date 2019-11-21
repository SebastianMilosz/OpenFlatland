#include "serializable_property_multiple_selection.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::PropertyMultipleSelection( PropertyNode* prop )
    {
        if ( prop != nullptr )
        {
            m_selectionVector.push_back( prop );
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::~PropertyMultipleSelection()
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyMultipleSelection::Add( PropertyNode* prop )
    {
        if ( prop != nullptr )
        {
            m_selectionVector.push_back( prop );
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator=( const PropertyMultipleSelection& val )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyMultipleSelection::operator==(const PropertyMultipleSelection& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyMultipleSelection::operator!=(const PropertyMultipleSelection& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyMultipleSelection::operator==(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyMultipleSelection::operator!=(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator=(bool val)
    {
        for ( auto* propSelection : m_selectionVector )
        {
            *propSelection = val;
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator=(char val)
    {
        for ( auto* propSelection : m_selectionVector )
        {
            *propSelection = val;
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator=(unsigned char val)
    {
        for ( auto* propSelection : m_selectionVector )
        {
            *propSelection = val;
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator=(int val)
    {
        for (auto* propSelection : m_selectionVector)
        {
            *propSelection = val;
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator=(unsigned int val)
    {
        for ( auto* propSelection : m_selectionVector )
        {
            *propSelection = val;
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator=(float val)
    {
        for ( auto* propSelection : m_selectionVector )
        {
            *propSelection = val;
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator=(double val)
    {
        for ( auto* propSelection : m_selectionVector )
        {
            *propSelection = val;
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator=( const std::string& val )
    {
        for ( auto* propSelection : m_selectionVector )
        {
            *propSelection = val;
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator++()
    {
        for ( auto* propSelection : m_selectionVector )
        {
            ++(*propSelection);
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator--()
    {
        for ( auto* propSelection : m_selectionVector )
        {
            --(*propSelection);
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator+=( const PropertyMultipleSelection& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator-=( const PropertyMultipleSelection& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator+(const PropertyMultipleSelection& rhs)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator-( const PropertyMultipleSelection& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyMultipleSelection::Name() const
    {
        std::string name;

        for ( auto* propSelection : m_selectionVector )
        {
            name += propSelection->Name();
        }

        return name;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyMultipleSelection::NameIs( const std::string& name ) const
    {
        std::string thisName( Name() );
        if ( thisName.compare( name ) == 0 )
        {
            return true;
        }
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyMultipleSelection::ConnectReference( smart_ptr<PropertyNode> refNode )
    {
        return false;
    }
}
