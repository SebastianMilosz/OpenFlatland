#include "serializable_property_selection.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::PropertySelection( PropertyNode* prop )
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
    PropertySelection::~PropertySelection()
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertySelection::Add( PropertyNode* prop )
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
    PropertyNode& PropertySelection::operator=( const PropertySelection& val )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertySelection::operator==(const PropertySelection& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertySelection::operator!=(const PropertySelection& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertySelection::operator==(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertySelection::operator!=(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(bool val)
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
    PropertyNode& PropertySelection::operator=(char val)
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
    PropertyNode& PropertySelection::operator=(unsigned char val)
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
    PropertyNode& PropertySelection::operator=(int val)
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
    PropertyNode& PropertySelection::operator=(unsigned int val)
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
    PropertyNode& PropertySelection::operator=(float val)
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
    PropertyNode& PropertySelection::operator=(double val)
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
    PropertyNode& PropertySelection::operator=( const std::string& val )
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
    PropertyNode& PropertySelection::operator++()
    {
        for (auto* propSelection : m_selectionVector)
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
    PropertyNode& PropertySelection::operator--()
    {
        for (auto* propSelection : m_selectionVector)
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
    PropertyNode& PropertySelection::operator+=( const PropertySelection& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator-=( const PropertySelection& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator+(const PropertySelection& rhs)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator-( const PropertySelection& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertySelection::Name() const
    {
        std::string name;

        for (auto* propSelection : m_selectionVector)
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
    bool PropertySelection::NameIs( const std::string& name ) const
    {
        std::string thisName = Name();
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
    bool PropertySelection::ConnectReference( smart_ptr<PropertyNode> refNode )
    {

    }
}
