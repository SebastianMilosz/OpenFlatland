#ifndef SERIALIZABLE_PROPERTY_SELECTION_HPP_INCLUDED
#define SERIALIZABLE_PROPERTY_SELECTION_HPP_INCLUDED

#include "serializable_property_node.hpp"

#include <string>
#include <vector>

namespace codeframe
{
     template <typename PROPERTY_TYPE> class Property;

     /*****************************************************************************
     * @class This class stores Property's from selection
     *****************************************************************************/
    class PropertySelection : public PropertyNode, public sigslot::has_slots<>
    {
        public:
            PropertySelection( PropertyNode* prop );
            PropertySelection( const PropertySelection& prop );
           ~PropertySelection();

            std::string   Name() const override;
            bool_t        NameIs( const std::string& name ) const override;
            std::string   ToString() const override;
            eType         Type() const override;
            std::string   Path( bool_t addName = true ) const;
            smart_ptr<PropertyNode> Reference() const override;
            uint32_t      Id() const override;

            smart_ptr<ObjectNode> Parent() const override;
            std::string ParentName() const override;
            bool_t ConnectReference( smart_ptr<PropertyNode> refNode ) override;
            std::string TypeString() const override;

            std::string PreviousValueString() const override;
            std::string CurentValueString() const override;
            int         PreviousValueInteger() const override;
            int         CurentValueInteger() const override;

            template<typename T>
            const T& GetConstValue() const
            {
                if (m_selection)
                {
                    auto propertyObject = dynamic_cast< codeframe::Property<T>* >(m_selection);
                    if (propertyObject)
                    {
                        return propertyObject->GetConstValue();
                    }
                }
                static T dummyValue;
                return dummyValue;
            }

            template<typename T>
            T& GetValue()
            {
                if (m_selection)
                {
                    auto propertyObject = dynamic_cast< codeframe::Property<T>* >(m_selection);
                    if (propertyObject)
                    {
                        return propertyObject->GetValue();
                    }
                }
                static T dummyValue;
                return dummyValue;
            }

            bool_t operator==(const int& sval) const override;
            bool_t operator!=(const int& sval) const override;

            PropertyNode& operator=(const bool_t        val) override;
            PropertyNode& operator=(const char          val) override;
            PropertyNode& operator=(const unsigned char val) override;
            PropertyNode& operator=(const int           val) override;
            PropertyNode& operator=(const unsigned int  val) override;
            PropertyNode& operator=(const float         val) override;
            PropertyNode& operator=(const double        val) override;
            PropertyNode& operator=(const std::string&  val) override;
            PropertyNode& operator++() override;
            PropertyNode& operator--() override;
            PropertyNode& operator+=(const int rhs) override;
            PropertyNode& operator-=(const int rhs) override;

            bool_t        operator==(const PropertyNode& sval) const;
            bool_t        operator!=(const PropertyNode& sval) const;
            PropertyNode& operator+=(const PropertyNode& rhs) override;
            PropertyNode& operator-=(const PropertyNode& rhs) override;
            PropertyNode& operator+ (const PropertyNode& rhs) override;
            PropertyNode& operator- (const PropertyNode& rhs) override;

            PropertySelection& operator =(const PropertySelection& rhs);

            operator bool() const override;
            operator char() const override;
            operator unsigned char() const override;
            operator int() const override;
            operator unsigned int() const override;
            operator unsigned short() const override;
            operator double() const override;
            operator float() const override;
            operator std::string() const override;

            bool_t      IsReference() const override;
            int         ToInt() const override;
            void        SetNumber( const int val ) override;
            int         GetNumber() const override;
            void        SetReal( const double val ) override;
            double      GetReal() const override;
            void        SetString( const std::string&  val ) override;
            std::string GetString() const override;

            void Lock() const override;
            void Unlock() const override;

            bool_t IsChanged() const override;

        private:
            void EmitChanges() override;
            void OnDelete(void* deletedPtr);

            PropertyNode* m_selection;
    };
}

#endif // SERIALIZABLE_PROPERTY_SELECTION_HPP_INCLUDED
