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
    };

    enum eXMLMode
    {
        XMLMODE_NON = 0x00,
        XMLMODE_R = 0x01,
        XMLMODE_W = 0x02,
        XMLMODE_RW = 0x03
    };

    /*****************************************************************************
    * @class cPropertyInfo
    * @brief Klasa zawiera wszystkie ustawienia konfiguracyjne klasy Property
    * dostêp do nich jest potokowy czyli: cPropertyInfo(this).Config1(a).Config2(b)
    *****************************************************************************/
    class cPropertyInfo
    {
        friend class PropertyBase;

        private:
            void Init();

            std::string             m_description;
            eKind                   m_kind;
            std::string             m_enumArray;
            bool                    m_eventEnable;
            int                     m_min;
            int                     m_max;
            bool                    m_enable;
            cRegister               m_register;
            eXMLMode                m_xmlmode;
            ReferenceManager        m_refmgr;
            PropertyBase*           m_serializableProperty;

        public:
            cPropertyInfo();
            cPropertyInfo(const cPropertyInfo& sval);
            cPropertyInfo(const cPropertyInfo& sval, PropertyBase* serializableProperty );
            cPropertyInfo& Register   ( eREG_MODE mod,
                                        uint16_t  reg,
                                        uint16_t  regSize = 1,
                                        uint16_t  cellOffset = 0,
                                        uint16_t  cellSize = 1,
                                        uint16_t  bitMask = 0xFFFF
                                      );
            cPropertyInfo& Description  ( const std::string& desc );
            cPropertyInfo& Kind         ( eKind kind );
            cPropertyInfo& Enum         ( const std::string& enuma );
            cPropertyInfo& ReferencePath( const std::string& referencePath );
            cPropertyInfo& Event        ( int e );
            cPropertyInfo& Min          ( int min );
            cPropertyInfo& Max          ( int max );
            cPropertyInfo& Enable       ( int state );
            cPropertyInfo& XMLMode      ( eXMLMode mode );

            // Accessors
            inline cRegister&         GetRegister()            { return m_register;     }
            inline eKind              GetKind()          const { return m_kind;         }
            inline eXMLMode           GetXmlMode()       const { return m_xmlmode;      }
            inline const std::string& GetDescription()   const { return m_description;  }
            inline const std::string& GetEnum()          const { return m_enumArray;    }
            inline const std::string& GetReferencePath() const { return m_refmgr.Get(); }
            inline bool               IsEventEnable()    const { return m_eventEnable;  }
            inline int                GetMin()           const { return m_min;          }
            inline int                GetMax()           const { return m_max;          }
            inline bool               GetEnable()        const { return m_enable;       }

            // Operators
            cPropertyInfo& operator=(cPropertyInfo val);
    };
}

#endif // SERIALIZABLE_PROPERTY_BASE_HPP_INCLUDED
