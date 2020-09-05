#ifndef SERIALIZABLE_PROPERTY_INFO_HPP_INCLUDED
#define SERIALIZABLE_PROPERTY_INFO_HPP_INCLUDED

#include <map>
#include <string>
#include <limits.h>

#include "reference_manager.hpp"
#include "serializable_register.hpp"

namespace codeframe
{
    enum eKind
    {
        KIND_NON = 0,
        KIND_LOGIC,
        KIND_NUMBER,
        KIND_NUMBERRANGE,
        KIND_REAL,
        KIND_TEXT,
        KIND_ENUM,
        KIND_DIR,
        KIND_URL,
        KIND_FILE,
        KIND_DATE,
        KIND_FONT,
        KIND_COLOR,
        KIND_IMAGE,
        KIND_2DPOINT,
        KIND_VECTOR,
        KIND_VECTOR_THRUST_HOST,
        KIND_RAY_DATA
    };

    enum eDepth
    {
        DEPTH_EXTERNAL_KIND = 0U,
        DEPTH_INTERNAL_KIND = 1U,
    };

    enum eXMLMode
    {
        XMLMODE_NON = 0x00,
        XMLMODE_R = 0x01,
        XMLMODE_W = 0x02,
        XMLMODE_RW = 0x03
    };

    enum eGUIMode
    {
        GUIMODE_NON = 0x00,
        GUIMODE_DISABLED = 0x01,
    };

    /*****************************************************************************
    * @class cPropertyInfo
    * @brief cPropertyInfo(this).Config1(a).Config2(b)
    *****************************************************************************/
    class cPropertyInfo
    {
        friend class PropertyBase;

        public:
            cPropertyInfo();
            cPropertyInfo(const cPropertyInfo& sval);
            cPropertyInfo(const cPropertyInfo& sval, PropertyBase* serializableProperty );
            cPropertyInfo& Register   ( eREG_MODE mod,
                                        uint16_t  reg,
                                        uint16_t  regSize = 1U,
                                        uint16_t  cellOffset = 0U,
                                        uint16_t  cellSize = 1U,
                                        uint16_t  bitMask = 0xFFFFU
                                      );
            cPropertyInfo& Description  ( const std::string& desc );
            cPropertyInfo& Kind         ( eKind kind1, eKind kind2=KIND_NON );
            cPropertyInfo& Enum         ( const std::string& enuma );
            cPropertyInfo& ReferencePath( const std::string& referencePath );
            cPropertyInfo& Event        ( int e );
            cPropertyInfo& Min          ( int min );
            cPropertyInfo& Max          ( int max );
            cPropertyInfo& Enable       ( int state );
            cPropertyInfo& XMLMode      ( eXMLMode mode );
            cPropertyInfo& GUIMode      ( eGUIMode mode );

            // Accessors
            cRegister&         GetRegister();
            eKind              GetKind(uint8_t depth = DEPTH_EXTERNAL_KIND) const;
            eXMLMode           GetXmlMode() const;
            const std::string& GetDescription() const;
            const std::string& GetEnum() const;
            const std::string& GetReferencePath() const;
            bool               IsEventEnable() const;
            int                GetMin() const;
            int                GetMax() const;
            bool               GetEnable() const;
            bool               GetGuiEnable() const;

            // Operators
            cPropertyInfo& operator=(cPropertyInfo val);

        private:
            static constexpr uint8_t KIND_DEPTH = 2U;

            void Init();

            std::string             m_description;
            eKind                   m_kind[KIND_DEPTH];
            std::string             m_enumArray;
            bool                    m_eventEnable;
            int                     m_min;
            int                     m_max;
            bool                    m_enable;
            cRegister               m_register;
            eXMLMode                m_xmlmode;
            eGUIMode                m_guimode;
            ReferenceManager        m_refmgr;
            PropertyBase*           m_serializableProperty;
    };

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline cRegister& cPropertyInfo::GetRegister()
    {
        return m_register;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline eKind cPropertyInfo::GetKind(uint8_t depth) const
    {
        return (depth < KIND_DEPTH) ? m_kind[depth] : KIND_NON;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline eXMLMode cPropertyInfo::GetXmlMode() const
    {
        return m_xmlmode;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline const std::string& cPropertyInfo::GetDescription() const
    {
        return m_description;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline const std::string& cPropertyInfo::GetEnum() const
    {
        return m_enumArray;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline const std::string& cPropertyInfo::GetReferencePath() const
    {
        return m_refmgr.Get();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline bool cPropertyInfo::IsEventEnable() const
    {
        return m_eventEnable;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline int cPropertyInfo::GetMin() const
    {
        return m_min;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline int cPropertyInfo::GetMax() const
    {
        return m_max;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline bool cPropertyInfo::GetEnable() const
    {
        return m_enable;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    inline bool cPropertyInfo::GetGuiEnable() const
    {
        return (m_guimode & GUIMODE_DISABLED);
    }
}

#endif // SERIALIZABLE_PROPERTY_BASE_HPP_INCLUDED
