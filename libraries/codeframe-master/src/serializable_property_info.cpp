#include "serializable_property_info.hpp"

namespace codeframe
{
/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cPropertyInfo::Init()
{
    m_description     = "";
    m_kind[0U]        = KIND_NON;
    m_kind[1U]        = KIND_NON;
    m_xmlmode         = XMLMODE_RW;
    m_guimode         = GUIMODE_NON;
    m_eventEnable     = true;
    m_eventValue      = 0U;
    m_min             = INT_MIN;
    m_max             = INT_MAX;
    m_enable          = true;
    m_visibleRange  = 0U;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo::cPropertyInfo() :
    m_refmgr(),
    m_serializableProperty(NULL)
{
    Init();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo::cPropertyInfo(const cPropertyInfo& sval) :
    m_description(sval.m_description),
    m_enumArray(sval.m_enumArray),
    m_eventEnable(sval.m_eventEnable),
    m_eventValue(sval.m_eventValue),
    m_min(sval.m_min),
    m_max(sval.m_max),
    m_enable(sval.m_enable) ,
    m_register(sval.m_register),
    m_xmlmode(sval.m_xmlmode),
    m_guimode(sval.m_guimode),
    m_refmgr(sval.m_refmgr),
    m_serializableProperty(sval.m_serializableProperty),
    m_visibleRange(sval.m_visibleRange)
{
    m_kind[0U] = sval.m_kind[0U];
    m_kind[1U] = sval.m_kind[1U];
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo::cPropertyInfo(const cPropertyInfo& sval, PropertyBase* serializableProperty ) :
    m_description(sval.m_description),
    m_enumArray(sval.m_enumArray),
    m_eventEnable(sval.m_eventEnable),
    m_eventValue(sval.m_eventValue),
    m_min(sval.m_min),
    m_max(sval.m_max),
    m_enable(sval.m_enable) ,
    m_register(sval.m_register),
    m_xmlmode(sval.m_xmlmode),
    m_guimode(sval.m_guimode),
    m_refmgr(sval.m_refmgr),
    m_serializableProperty( serializableProperty ),
    m_visibleRange(sval.m_visibleRange)
{
    m_kind[0U] = sval.m_kind[0U];
    m_kind[1U] = sval.m_kind[1U];
    m_refmgr.SetProperty( m_serializableProperty );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::Register( eREG_MODE mod,
                                         uint16_t  reg,
                                         uint16_t  regSize,
                                         uint16_t  cellOffset,
                                         uint16_t  cellSize,
                                         uint16_t  bitMask
                                       )
{
    m_register.Set( mod, reg, regSize, cellOffset, cellSize, bitMask);
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::Description( const std::string& desc )
{
    m_description = desc;
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::Kind( eKind kind1, eKind kind2 )
{
    m_kind[0] = kind1;
    m_kind[1] = kind2;

    m_visibleRange = 0U;

    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::Kind( eKind kind1, uint8_t visibleRange )
{
    m_kind[0] = kind1;
    m_kind[1] = KIND_NON;

    m_visibleRange = visibleRange;

    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::Enum( const std::string& enuma )
{
    m_enumArray = enuma;
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::ReferencePath( const std::string& referencePath )
{
    m_refmgr.SetReference( referencePath, m_serializableProperty );
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::Event( int e )
{
    m_eventEnable = e;
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::EventValue( uint32_t value )
{
    m_eventValue = value;
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::Min( int min )
{
    m_min = min;
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::Max( int max )
{
    m_max = max;
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::Enable( int state )
{
    m_enable = state;
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::XMLMode( eXMLMode mode )
{
    m_xmlmode = mode;
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::GUIMode( eGUIMode mode )
{
    m_guimode = mode;
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo& cPropertyInfo::operator=(cPropertyInfo val)
{
    m_description           = val.m_description;
    m_kind[0U]              = val.m_kind[0U];
    m_kind[1U]              = val.m_kind[1U];
    m_xmlmode               = val.m_xmlmode;
    m_guimode               = val.m_guimode;
    m_enumArray	            = val.m_enumArray;
    m_register              = val.m_register;
    m_eventEnable           = val.m_eventEnable;
    m_eventValue            = val.m_eventValue;
    m_min                   = val.m_min;
    m_max                   = val.m_max;
    m_enable                = val.m_enable;
    m_refmgr                = val.m_refmgr;
    m_serializableProperty  = val.m_serializableProperty;
    return *this;
}

}
