#include "serializable_property_multiple_selection.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::PropertyMultipleSelection()
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::PropertyMultipleSelection( const PropertyMultipleSelection& prop )
    {
        m_selectionVector = prop.m_selectionVector;
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
    void PropertyMultipleSelection::Add( smart_ptr<PropertyNode> prop )
    {
        if ( smart_ptr_isValid( prop ) )
        {
            m_selectionVector.push_back( prop );
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator=( const PropertyNode& val )
    {
        const PropertyMultipleSelection* multiSel = dynamic_cast<const PropertyMultipleSelection*>(&val);
        if ( (const PropertyMultipleSelection*)nullptr != multiSel )
        {
            m_selectionVector = multiSel->m_selectionVector;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection& PropertyMultipleSelection::operator=(const PropertyMultipleSelection& rhs)
    {
        m_selectionVector = rhs.m_selectionVector;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyMultipleSelection::operator==(const PropertyNode& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyMultipleSelection::operator!=(const PropertyNode& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyMultipleSelection::operator==(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyMultipleSelection::operator!=(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator=(const bool_t val)
    {
        for ( auto propSelection : m_selectionVector )
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
    PropertyNode& PropertyMultipleSelection::operator=(const char val)
    {
        for ( auto propSelection : m_selectionVector )
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
    PropertyNode& PropertyMultipleSelection::operator=(const unsigned char val)
    {
        for ( auto propSelection : m_selectionVector )
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
    PropertyNode& PropertyMultipleSelection::operator=(const int val)
    {
        for (auto propSelection : m_selectionVector)
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
    PropertyNode& PropertyMultipleSelection::operator=(const unsigned int val)
    {
        for ( auto propSelection : m_selectionVector )
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
    PropertyNode& PropertyMultipleSelection::operator=(const float val)
    {
        for ( auto propSelection : m_selectionVector )
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
    PropertyNode& PropertyMultipleSelection::operator=(const double val)
    {
        for ( auto propSelection : m_selectionVector )
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
        for ( auto propSelection : m_selectionVector )
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
    PropertyNode& PropertyMultipleSelection::operator+=(const int rhs)
    {
        for ( auto propSelection : m_selectionVector )
        {
            (*propSelection) += rhs;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator-=(const int rhs)
    {
        for ( auto propSelection : m_selectionVector )
        {
            (*propSelection) -= rhs;
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
        for ( auto propSelection : m_selectionVector )
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
        for ( auto propSelection : m_selectionVector )
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
    PropertyNode& PropertyMultipleSelection::operator+=( const PropertyNode& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator-=( const PropertyNode& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator+(const PropertyNode& rhs)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyMultipleSelection::operator-( const PropertyNode& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::operator bool() const
    {
        bool retVal = true;
        for ( auto propSelection : m_selectionVector )
        {
            retVal &= (bool)(*propSelection);
        }
        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::operator char() const
    {
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::operator unsigned char() const
    {
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::operator int() const
    {
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::operator unsigned int() const
    {
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::operator unsigned short() const
    {
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::operator double() const
    {
        return 0.0F;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::operator float() const
    {
        return 0.0F;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyMultipleSelection::operator std::string() const
    {
        std::string retVal;
        for ( auto propSelection : m_selectionVector )
        {
            retVal += (std::string)(*propSelection);
            retVal += " ";
        }
        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyMultipleSelection::IsReference() const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertyMultipleSelection::ToInt() const
    {
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyMultipleSelection::SetNumber( const int val )
    {
        for ( auto propSelection : m_selectionVector )
        {
            *propSelection = val;
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertyMultipleSelection::GetNumber() const
    {
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyMultipleSelection::SetReal( const double val )
    {
        for ( auto propSelection : m_selectionVector )
        {
            *propSelection = val;
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    double PropertyMultipleSelection::GetReal() const
    {
        return 0.0F;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyMultipleSelection::SetString( const std::string&  val )
    {
        for ( auto propSelection : m_selectionVector )
        {
            *propSelection = val;
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyMultipleSelection::GetString() const
    {
        return (std::string)(*this);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyMultipleSelection::Name() const
    {
        std::string name;

        for ( auto propSelection : m_selectionVector )
        {
            name += propSelection->Name();
            name += " ";
        }

        return name;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyMultipleSelection::NameIs( const std::string& name ) const
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
    std::string PropertyMultipleSelection::ToString() const
    {
        std::string retVal;

        for ( auto propSelection : m_selectionVector )
        {
            retVal += propSelection->ToString();
            retVal += " ";
        }

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    eType PropertyMultipleSelection::Type() const
    {
        return eType::TYPE_VECTOR;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyMultipleSelection::Path( bool_t addName ) const
    {
        return "Multiple::Selection::Path";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<PropertyNode> PropertyMultipleSelection::Reference() const
    {
        return nullptr;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    uint32_t PropertyMultipleSelection::Id() const
    {
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    ObjectNode* PropertyMultipleSelection::Parent() const
    {
        return nullptr;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyMultipleSelection::ParentName() const
    {
        std::string retVal;

        for ( auto propSelection : m_selectionVector )
        {
            retVal += propSelection->ParentName();
            retVal += " ";
        }

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyMultipleSelection::ConnectReference( smart_ptr<PropertyNode> refNode )
    {
        bool_t retVal = false;

        for ( auto propSelection : m_selectionVector )
        {
            retVal |= propSelection->ConnectReference( refNode );
        }

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyMultipleSelection::TypeString() const
    {
        std::string retVal = "";

        for ( auto propSelection : m_selectionVector )
        {
            retVal += propSelection->TypeString();
            retVal += " ";
        }

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyMultipleSelection::PreviousValueString() const
    {
        std::string retVal = "";

        for ( auto propSelection : m_selectionVector )
        {
            retVal += propSelection->PreviousValueString();
            retVal += " ";
        }

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyMultipleSelection::CurentValueString() const
    {
        std::string retVal = "";

        for ( auto propSelection : m_selectionVector )
        {
            retVal += propSelection->CurentValueString();
            retVal += " ";
        }

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertyMultipleSelection::PreviousValueInteger() const
    {
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertyMultipleSelection::CurentValueInteger() const
    {
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyMultipleSelection::Lock() const
    {
        for ( auto propSelection : m_selectionVector )
        {
            propSelection->Lock();
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyMultipleSelection::Unlock() const
    {
        for ( auto propSelection : m_selectionVector )
        {
            propSelection->Unlock();
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyMultipleSelection::IsChanged() const
    {
        bool_t retVal = false;

        for ( auto propSelection : m_selectionVector )
        {
            retVal |= propSelection->IsChanged();
        }

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyMultipleSelection::EmitChanges()
    {
        for ( auto propSelection : m_selectionVector )
        {
            propSelection->EmitChanges();
        }
    }
}
