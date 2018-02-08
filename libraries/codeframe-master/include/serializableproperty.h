#ifndef _SERIALIZABLEPROPERTY_H
#define _SERIALIZABLEPROPERTY_H

#include "serializableregister.h"

#include <ThreadUtilities.h>
#include <sigslot.h>
#include <climits>

#ifdef SERIALIZABLE_USE_CIMAGE
#include <cimage.h>
#endif

namespace codeframe
{
    enum eType    { TYPE_NON, TYPE_CHAR, TYPE_INT, TYPE_REAL, TYPE_TEXT, TYPE_IMAGE };
    enum eKind    { KIND_NON, KIND_LOGIC, KIND_NUMBER, KIND_NUMBERRANGE, KIND_REAL, KIND_TEXT, KIND_ENUM, KIND_DIR, KIND_URL, KIND_FILE, KIND_DATE, KIND_FONT, KIND_COLOR, KIND_IMAGE };
    enum eXMLMode { XMLMODE_NON = 0x00, XMLMODE_R = 0x01, XMLMODE_W = 0x02, XMLMODE_RW = 0x03 };

    using namespace sigslot;

    class cSerializable;

    /*****************************************************************************
    * @class cPropertyInfo
    * @brief Klasa zawiera wszystkie ustawienia konfiguracyjne klasy Property
    * dostęp do nich jest potokowy czyli: cPropertyInfo(this).Config1(a).Config2(b)
    *****************************************************************************/
    class cPropertyInfo
    {
        friend class Property;

        private:
            std::string m_description;
            eKind       m_kind;
            std::string m_enumArray;
            int         m_imageWidth;
            int         m_imageHeight;
            bool        m_eventEnable;
            int         m_min;
            int         m_max;
            bool        m_enable;
            cRegister   m_register;
            eXMLMode    m_xmlmode;

            void Init()
            {
                m_description   = "";
                m_kind          = KIND_NON;
                m_xmlmode       = XMLMODE_RW;
                m_enumArray     = "";
                m_imageWidth    = 640;
                m_imageHeight   = 480;
                m_eventEnable   = true;
                m_min           = INT_MIN;
                m_max           = INT_MAX;
                m_enable        = true;
            }

        public:
            cPropertyInfo() { Init(); }
            cPropertyInfo(const cPropertyInfo& sval) :
            m_description(sval.m_description),
            m_kind(sval.m_kind),
            m_enumArray(sval.m_enumArray),
            m_imageWidth(sval.m_imageWidth),
            m_imageHeight(sval.m_imageHeight),
            m_eventEnable(sval.m_eventEnable),
            m_min(sval.m_min),
            m_max(sval.m_max),
            m_enable(sval.m_enable) ,
            m_register(sval.m_register),
            m_xmlmode(sval.m_xmlmode)
            {

            }
            cPropertyInfo& Register   ( eREG_MODE mod,
                                        uint16_t  reg,
                                        uint16_t  regSize = 1,
                                        uint16_t  cellOffset = 0,
                                        uint16_t  cellSize = 1,
                                        uint16_t  bitMask = 0xFFFF
                                      ) { m_register.Set( mod, reg, regSize, cellOffset, cellSize, bitMask);  return *this; }
            cPropertyInfo& Description( std::string desc                        ) { m_description = desc;     return *this; }
            cPropertyInfo& Kind       ( eKind       kind                        ) { m_kind        = kind;     return *this; }
            cPropertyInfo& Enum       ( std::string enuma                       ) { m_enumArray   = enuma;    return *this; }
            cPropertyInfo& Width      ( int         w                           ) { m_imageWidth  = w;        return *this; }
            cPropertyInfo& Height     ( int         h                           ) { m_imageHeight = h;        return *this; }
            cPropertyInfo& Event      ( int         e                           ) { m_eventEnable = e;        return *this; }
            cPropertyInfo& Min        ( int         min                         ) { m_min = min;              return *this; }
            cPropertyInfo& Max        ( int         max                         ) { m_max = max;              return *this; }
            cPropertyInfo& Enable     ( int         state                       ) { m_enable = state;         return *this; }
            cPropertyInfo& XMLMode    ( eXMLMode    mode                        ) { m_xmlmode = mode;         return *this;}


            // Accessors
            cRegister&  GetRegister()          { return m_register;    }
            eKind 	    GetKind()        const { return m_kind;        }
            eXMLMode    GetXmlMode()     const { return m_xmlmode;     }
            std::string GetDescription() const { return m_description; }
            std::string GetEnum()        const { return m_enumArray;   }
            int         GetWidth()       const { return m_imageWidth;  }
            int         GetHeight()      const { return m_imageHeight; }
            bool        IsEventEnable()  const { return m_eventEnable; }
            int         GetMin()         const { return m_min;         }
            int         GetMax()         const { return m_max;         }
            bool        GetEnable()      const { return m_enable;      }

            // Operators
            cPropertyInfo& operator=(cPropertyInfo val)
            {
                m_description   = val.m_description;
                m_kind          = val.m_kind;
                m_xmlmode       = val.m_xmlmode;
                m_enumArray	    = val.m_enumArray;
                m_imageWidth    = val.m_imageWidth;
                m_imageHeight   = val.m_imageHeight;
                m_register      = val.m_register;
                m_eventEnable   = val.m_eventEnable;
                m_min           = val.m_min;
                m_max           = val.m_max;
                m_enable        = val.m_enable;
                return *this;
            }
    };

    /*****************************************************************************
     * @class Property
     *****************************************************************************/
    class Property
    {
        friend class cPropertyInfo;

        protected:
            static int     s_globalParConCnt;
            Property*      m_reference;             ///< Wskaznik na sprzezone z tym polem pole
            cSerializable* m_referenceParent;
            eType          m_type;
            cSerializable* m_parentpc;
            std::string    m_name;
            uint32_t       m_id;
            WrMutex        m_Mutex;
            bool           m_isWaitForUpdate;
            int            m_waitForUpdateCnt;
            cPropertyInfo  m_propertyInfo;
            bool           m_pulseAbort;

            // Typy trywialne
            union uType
            {
                char    type_char;
                int     type_int;
                double  type_real;
            } v, prew_v;

            // Typy nie trywialne
            std::string prew_type_text;
            std::string type_text;

            bool m_temporary;

            void     RegisterProperty();
            void	 UnRegisterProperty();
            void     ValueUpdate();

            static uint32_t GetHashId(std::string str, uint16_t mod = 0 );

        public:
            // Konstruktory typow
            Property( cSerializable* parentpc, std::string name, eType type, cPropertyInfo info ) :
                m_reference(NULL),
                m_referenceParent(NULL),
                m_type(type),
                m_parentpc(parentpc),
                m_name(name),
                m_id(0),
                m_isWaitForUpdate(false),
                m_waitForUpdateCnt(0),
                m_propertyInfo( info ),
                m_pulseAbort(false),
                prew_type_text(""),
                type_text(""),
                m_temporary( false )
                {
                    v.type_real      = 0;
                    prew_v.type_real = 0;
                    RegisterProperty();
                }
        protected:
            Property( cSerializable* parentpc, std::string name, bool val, cPropertyInfo info ) :
                m_reference(NULL),
                m_referenceParent(NULL),
                m_type(TYPE_CHAR),
                m_parentpc(parentpc),
                m_name(name),
                m_id(0),
                m_isWaitForUpdate(false),
                m_waitForUpdateCnt(0),
                m_propertyInfo( info ),
                m_pulseAbort(false),
                prew_type_text(""),
                type_text(""),
                m_temporary( false )
                {
                    v.type_real      = 0;
                    prew_v.type_real = 0;
                    prew_v.type_char = (bool)val;
                    v.type_char      = (bool)val;
                    RegisterProperty();
                }
            Property( cSerializable* parentpc, std::string name, char val, cPropertyInfo info ) :
                m_reference(NULL),
                m_referenceParent(NULL),
                m_type(TYPE_CHAR),
                m_parentpc(parentpc),
                m_name(name),
                m_id(0),
                m_isWaitForUpdate(false),
                m_waitForUpdateCnt(0),
                m_propertyInfo( info ),
                m_pulseAbort(false),
                prew_type_text(""),
                type_text(""),
                m_temporary( false )
                {
                    v.type_real      = 0;
                    prew_v.type_real = 0;
                    prew_v.type_char = val;
                    v.type_char      = val;
                    RegisterProperty();
                }
            Property( cSerializable* parentpc, std::string name, int val, cPropertyInfo info ) :
                m_reference(NULL),
                m_referenceParent(NULL),
                m_type(TYPE_INT),
                m_parentpc(parentpc),
                m_name(name),
                m_id(0),
                m_isWaitForUpdate(false),
                m_waitForUpdateCnt(0),
                m_propertyInfo( info ),
                m_pulseAbort(false),
                prew_type_text(""),
                type_text(""),
                m_temporary( false )
                {
                    v.type_real      = 0;
                    prew_v.type_real = 0;
                    prew_v.type_int  = val;
                    v.type_int       = val;
                    RegisterProperty();
                }
            Property( cSerializable* parentpc, std::string name, double val, cPropertyInfo info ) :
                m_reference(NULL),
                m_referenceParent(NULL),
                m_type(TYPE_REAL),
                m_parentpc(parentpc),
                m_name(name),
                m_id(0),
                m_isWaitForUpdate(false),
                m_waitForUpdateCnt(0),
                m_propertyInfo( info ),
                m_pulseAbort(false),
                prew_type_text(""),
                type_text(""),
                m_temporary( false )
                {
                    v.type_real      = val;
                    prew_v.type_real = val;
                    RegisterProperty();
                }
            Property( cSerializable* parentpc, std::string name, std::string val, cPropertyInfo info ) :
                m_reference(NULL),
                m_referenceParent(NULL),
                m_type(TYPE_TEXT),
                m_parentpc(parentpc),
                m_name(name),
                m_id(0),
                m_isWaitForUpdate(false),
                m_waitForUpdateCnt(0),
                m_propertyInfo( info ),
                m_pulseAbort(false),
                prew_type_text(val),
                type_text(val),
                m_temporary( false )
                {
                    v.type_real      = 0;
                    prew_v.type_real = 0;
                    RegisterProperty();
                }
            Property( cSerializable* parentpc, std::string name, char* val, cPropertyInfo info ) :
                m_reference(NULL),
                m_referenceParent(NULL),
                m_type(TYPE_TEXT),
                m_parentpc(parentpc),
                m_name(name),
                m_id(0),
                m_isWaitForUpdate(false),
                m_waitForUpdateCnt(0),
                m_propertyInfo( info ),
                m_pulseAbort(false),
                prew_type_text(std::string(val)),
                type_text(std::string(val)),
                m_temporary( false )
                {
                    v.type_real      = 0;
                    prew_v.type_real = 0;
                    RegisterProperty();
                }

        public:
            virtual ~Property()
            {
                if( m_temporary == false )
                {
                    UnRegisterProperty();
                }
            }

            // Sygnaly
            signal1<Property*> signalChanged;

            // Konstruktor kopiujacy
            Property(const Property& sval) :
                m_reference      (sval.m_reference),
                m_referenceParent(sval.m_referenceParent),
                m_type           (sval.m_type),
                m_parentpc       (sval.m_parentpc),
                m_name           (sval.m_name),
                m_id             (sval.m_id),
                m_propertyInfo   (sval.m_propertyInfo),
                m_pulseAbort     (sval.m_pulseAbort),
                prew_type_text   (sval.prew_type_text),
                type_text        (sval.type_text),
                m_temporary      ( true )
            {
              switch( (int)sval.m_type )
              {
                case TYPE_INT:
                {
                  v.type_int      = sval.v.type_int;
                  prew_v.type_int = sval.prew_v.type_int;
                  break;
                }
                case TYPE_REAL:
                {
                  v.type_real      = sval.v.type_real;
                  prew_v.type_real = sval.prew_v.type_real;
                  break;
                }
                case TYPE_CHAR:
                {
                  v.type_char      = sval.v.type_char;
                  prew_v.type_char = sval.prew_v.type_char;
                  break;
                }
              }
            }

           bool IsReference() const;

            // Operator porownania
            bool operator==(const Property& sval);
            bool operator!=(const Property& sval);

            bool operator==(const int& sval);
            bool operator!=(const int& sval);

            // Operatory przypisania
            Property& operator=(Property     val);
            Property& operator=(bool         val);
            Property& operator=(char         val);
            Property& operator=(int          val);
            Property& operator=(unsigned int val);
            Property& operator=(double       val);
            Property& operator=(std::string  val);
            Property& operator++();
            Property  operator++(int);
            Property& operator--();
            Property  operator--(int);
            Property& operator+=(const Property& rhs);
            Property& operator-=(const Property& rhs);
            Property  operator+(const Property& rhs);
            Property  operator-(const Property& rhs);
            Property& operator+=(const int rhs);
            Property& operator-=(const int rhs);


            // Operatory rzutowania
            operator bool() const;
            operator char() const;
            operator int() const;
            operator unsigned int() const;
            operator unsigned short() const;
            operator double() const;
            operator std::string() const;

            int                     ToInt() const { return (int)(*this); }
            std::string             ToString();
            int                     ToEnumPosition( std::string enumStringValue );
            cPropertyInfo&          Info() { return m_propertyInfo; }
            virtual void        	WaitForUpdatePulse();
            virtual void         	WaitForUpdate(int time = 100);
            virtual std::string  	Name() const;
            virtual bool  	        NameIs( std::string name ) const;
            virtual uint32_t     	Id() const;
            virtual eType        	Type() const;
            virtual std::string  	Path(bool addName = true) const;
            virtual cSerializable* 	Parent() { return m_parentpc; }
            virtual Property*    	Reference() { return m_reference; }
            virtual bool         	ConnectReference( Property* refProp );
            virtual std::string  	TypeString() const;

            virtual std::string  	PreviousValueString() const;
            virtual std::string  	CurentValueString() const;
            virtual int             PreviousValueInteger() const;
            virtual int             CurentValueInteger() const;

            void                 	PulseChanged();
            void                    CommitChanges();
            bool                    IsChanged() const;
            Property&               WatchdogGetValue( int time = 1000 );

            // Geter/Seter LUA Interface
            void         	        SetNumber( int val );
            int                     GetNumber() const;
            void         	        SetReal( double val );
            double                  GetReal() const;
            void         	        SetString( std::string  val );
            std::string             GetString() const;
    };

    // Specjalizowane propertisy
    class Property_Int : public Property
    {
        public:
            Property_Int( cSerializable* parentpc, std::string name, int val,  cPropertyInfo info ) : Property( parentpc, name, val, info ) {}
           ~Property_Int() {}
           using Property::operator =;
    };

    class Property_Rea : public Property
    {
        public:
            Property_Rea( cSerializable* parentpc, std::string name, double val,  cPropertyInfo info ) : Property( parentpc, name, val, info ) {}
           ~Property_Rea() {}
           using Property::operator =;
    };

    class Property_Str : public Property
    {
        public:
            Property_Str( cSerializable* parentpc, std::string name, std::string val,  cPropertyInfo info ) : Property( parentpc, name, val, info ) {}
           ~Property_Str() {}
           using Property::operator =;
    };

    #ifdef SERIALIZABLE_USE_CIMAGE
    class Property_Img : public Property
    {
        private:
            cImage m_Image;

        public:
            Property_Img( cSerializable* parentpc, std::string name, cPropertyInfo info ) : Property( parentpc, name, TYPE_IMAGE, info )
            {
                int w = info.GetWidth();
                int h = info.GetHeight();
                m_Image.assign( w, h );
            }
           ~Property_Img() {}

           std::string TypeString() { return "image"; }

           void New(int w, int h)
           {
               m_Image.assign( w, h );
           }

           operator cImage&()
           {
               if( m_reference ) { return ( static_cast<Property_Img*>(m_reference) )->m_Image; }
               return m_Image;
           }
    };
    #endif

}

#endif
