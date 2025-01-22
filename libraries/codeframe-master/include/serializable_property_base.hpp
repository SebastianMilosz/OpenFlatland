#ifndef SERIALIZABLE_PROPERTY_BASE_HPP_INCLUDED
#define SERIALIZABLE_PROPERTY_BASE_HPP_INCLUDED

#include "serializable_register.hpp"
#include "serializable_property_info.hpp"
#include "serializable_property_node.hpp"
#include "serializable_property_selection.hpp"
#include "typeinfo.hpp"
#include "ThreadUtilities.h"
#include "DataTypesUtilities.h"

#include <climits>
#include <string>

namespace codeframe
{
    class ObjectNode;

    /*****************************************************************************
     * @class PropertyBase
     *****************************************************************************/
    class PropertyBase : public PropertyNode
    {
        friend class cPropertyInfo;

        public:
             PropertyBase( ObjectNode* parentpc, const std::string& name, eType type, cPropertyInfo info );
             PropertyBase( const PropertyBase& sval );
            ~PropertyBase() override;

            bool_t operator==(const int& sval) const override;
            bool_t operator!=(const int& sval) const override;

            void SetValue(const bool_t         val, bool triggerEvent = true) override;
            void SetValue(const char           val, bool triggerEvent = true) override;
            void SetValue(const unsigned char  val, bool triggerEvent = true) override;
            void SetValue(const int            val, bool triggerEvent = true) override;
            void SetValue(const unsigned int   val, bool triggerEvent = true) override;
            void SetValue(const short          val, bool triggerEvent = true) override;
            void SetValue(const unsigned short val, bool triggerEvent = true) override;
            void SetValue(const float          val, bool triggerEvent = true) override;
            void SetValue(const double         val, bool triggerEvent = true) override;
            void SetValue(const std::string&   val, bool triggerEvent = true) override;

            // Operatory przypisania
            PropertyNode& operator=(const bool_t         val) override;
            PropertyNode& operator=(const char           val) override;
            PropertyNode& operator=(const unsigned char  val) override;
            PropertyNode& operator=(const int            val) override;
            PropertyNode& operator=(const unsigned int   val) override;
            PropertyNode& operator=(const short          val) override;
            PropertyNode& operator=(const unsigned short val) override;
            PropertyNode& operator=(const float          val) override;
            PropertyNode& operator=(const double         val) override;
            PropertyNode& operator=(const std::string&   val) override;
            PropertyNode& operator++() override;
            PropertyNode& operator--() override;
            PropertyNode& operator+=(const int rhs) override;
            PropertyNode& operator-=(const int rhs) override;

            PropertyNode& operator=(const PropertyNode& rhs);
            PropertyNode& operator+=(const PropertyNode& rhs) override;
            PropertyNode& operator-=(const PropertyNode& rhs) override;
            PropertyNode& operator+(const PropertyNode& rhs) override;
            PropertyNode& operator-(const PropertyNode& rhs) override;
            bool_t        operator==(const PropertyBase& sval) const;
            bool_t        operator!=(const PropertyBase& sval) const;

            // Operatory rzutowania
            operator bool() const override;
            operator char() const override;
            operator unsigned char() const override;
            operator int() const override;
            operator unsigned int() const override;
            operator short() const override;
            operator unsigned short() const override;
            operator double() const override;
            operator float() const override;
            operator std::string() const override;

            bool_t      IsReference() const override;
            int         ToInt() const override { return (int)(*this); }
            std::string ToString() const override;
            int         ToEnumPosition( const std::string& enumStringValue ) const;
            void        WaitForUpdatePulse();
            void        WaitForUpdate( int time = 100 );
            std::string Name() const override;
            bool_t      NameIs( const std::string& name ) const override;
            uint32_t    Id() const override;
            eType       Type() const override;
            std::string Path(bool_t addName = true) const override;
            smart_ptr<ObjectNode> Parent() const override;
            std::string ParentName() const override;
            smart_ptr<PropertyNode> Reference() const override { return m_reference; }
            bool_t      ConnectReference( smart_ptr<PropertyNode> refNode ) override;
            std::string TypeString() const override;

            std::string PreviousValueString() const override;
            std::string CurentValueString() const override;
            int         PreviousValueInteger() const override;
            int         CurentValueInteger() const override;

            const cPropertyInfo& ConstInfo() const { return m_propertyInfo; }
            cPropertyInfo&       Info() { return m_propertyInfo; }

            void   PulseChanged();
            void   CommitChanges();
            bool_t IsChanged() const override;

            PropertyNode& WatchdogGetValue( int time = 1000 );

            void        SetNumber( const int val ) override;
            int         GetNumber() const override;
            void        SetReal( const double val ) override;
            double      GetReal() const override;
            void        SetString( const std::string&  val ) override;
            std::string GetString() const override;

            void Lock() const override;
            void Unlock() const override;

        protected:
            static int              s_globalParConCnt;
            smart_ptr<PropertyNode> m_reference;             ///< Reference to another property
            eType                   m_type;
            smart_ptr<ObjectNode>   m_parentpc;
            std::string             m_name;
            uint32_t                m_id;
            mutable WrMutex         m_Mutex;
            bool_t                  m_isWaitForUpdate;
            int                     m_waitForUpdateCnt;
            cPropertyInfo           m_propertyInfo;
            bool_t                  m_temporary;
            bool_t                  m_registered;

            void RegisterProperty();
            void UnRegisterProperty();
            void EmitChanges() override;
            void OnParentDelete(void* deletedPtr);

            static uint32_t GetHashId( const std::string& str, uint16_t mod = 0 );
    };
}

#endif // SERIALIZABLE_PROPERTY_BASE_HPP_INCLUDED
