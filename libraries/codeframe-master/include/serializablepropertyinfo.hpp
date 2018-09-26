#ifndef SERIALIZABLEPROPERTYINFO_HPP_INCLUDED
#define SERIALIZABLEPROPERTYINFO_HPP_INCLUDED

#include <map>
#include <string>
#include <limits.h>

#include "referencemanager.hpp"
#include "serializableregister.hpp"

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
            cSerializableInterface* m_serializableParent;

            void Init()
            {
                m_description   = "";
                m_kind          = KIND_NON;
                m_xmlmode       = XMLMODE_RW;
                m_eventEnable   = true;
                m_min           = INT_MIN;
                m_max           = INT_MAX;
                m_enable        = true;
            }

        public:
            cPropertyInfo() :
                m_refmgr(),
                m_serializableParent(NULL)
            {
                Init();
            }
            cPropertyInfo(const cPropertyInfo& sval) :
            m_description(sval.m_description),
            m_kind(sval.m_kind),
            m_enumArray(sval.m_enumArray),
            m_eventEnable(sval.m_eventEnable),
            m_min(sval.m_min),
            m_max(sval.m_max),
            m_enable(sval.m_enable) ,
            m_register(sval.m_register),
            m_xmlmode(sval.m_xmlmode),
            m_refmgr(sval.m_refmgr),
            m_serializableParent(sval.m_serializableParent)
            {

            }
            cPropertyInfo(const cPropertyInfo& sval, cSerializableInterface* serializableParent ) :
            m_description(sval.m_description),
            m_kind(sval.m_kind),
            m_enumArray(sval.m_enumArray),
            m_eventEnable(sval.m_eventEnable),
            m_min(sval.m_min),
            m_max(sval.m_max),
            m_enable(sval.m_enable) ,
            m_register(sval.m_register),
            m_xmlmode(sval.m_xmlmode),
            m_refmgr(sval.m_refmgr),
            m_serializableParent( serializableParent )
            {
                m_refmgr.SetParent( m_serializableParent );
            }
            cPropertyInfo& Register   ( eREG_MODE mod,
                                        uint16_t  reg,
                                        uint16_t  regSize = 1,
                                        uint16_t  cellOffset = 0,
                                        uint16_t  cellSize = 1,
                                        uint16_t  bitMask = 0xFFFF
                                      ) { m_register.Set( mod, reg, regSize, cellOffset, cellSize, bitMask);  return *this; }
            cPropertyInfo& Description  ( const std::string& desc ) { m_description = desc; return *this; }
            cPropertyInfo& Kind         ( eKind kind ) { m_kind = kind; return *this; }
            cPropertyInfo& Enum         ( const std::string& enuma ) { m_enumArray   = enuma; return *this; }
            cPropertyInfo& ReferencePath( const std::string& referencePath )
            {
                m_refmgr.SetReference( referencePath, m_serializableParent );
                return *this;
            }
            cPropertyInfo& Event        ( int         e             ) { m_eventEnable = e;        return *this; }
            cPropertyInfo& Min          ( int         min           ) { m_min = min;              return *this; }
            cPropertyInfo& Max          ( int         max           ) { m_max = max;              return *this; }
            cPropertyInfo& Enable       ( int         state         ) { m_enable = state;         return *this; }
            cPropertyInfo& XMLMode      ( eXMLMode    mode          ) { m_xmlmode = mode;         return *this;}


            // Accessors
            inline cRegister&  GetRegister()            { return m_register;    }
            inline eKind       GetKind()          const { return m_kind;        }
            inline eXMLMode    GetXmlMode()       const { return m_xmlmode;     }
            inline const std::string& GetDescription()   const { return m_description; }
            inline const std::string& GetEnum()          const { return m_enumArray;   }
            inline const std::string& GetReferencePath() const { return m_refmgr.Get(); }
            inline bool        IsEventEnable()    const { return m_eventEnable; }
            inline int         GetMin()           const { return m_min;         }
            inline int         GetMax()           const { return m_max;         }
            inline bool        GetEnable()        const { return m_enable;      }

            // Operators
            cPropertyInfo& operator=(cPropertyInfo val)
            {
                m_description        = val.m_description;
                m_kind               = val.m_kind;
                m_xmlmode            = val.m_xmlmode;
                m_enumArray	         = val.m_enumArray;
                m_register           = val.m_register;
                m_eventEnable        = val.m_eventEnable;
                m_min                = val.m_min;
                m_max                = val.m_max;
                m_enable             = val.m_enable;
                m_refmgr             = val.m_refmgr;
                m_serializableParent = val.m_serializableParent;
                return *this;
            }
    };
}

#endif // SERIALIZABLEPROPERTYINFO_HPP_INCLUDED
