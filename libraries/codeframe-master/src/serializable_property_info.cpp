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
    m_description   = "";
    m_kind          = KIND_NON;
    m_xmlmode       = XMLMODE_RW;
    m_eventEnable   = true;
    m_min           = INT_MIN;
    m_max           = INT_MAX;
    m_enable        = true;
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
    m_kind(sval.m_kind),
    m_enumArray(sval.m_enumArray),
    m_eventEnable(sval.m_eventEnable),
    m_min(sval.m_min),
    m_max(sval.m_max),
    m_enable(sval.m_enable) ,
    m_register(sval.m_register),
    m_xmlmode(sval.m_xmlmode),
    m_refmgr(sval.m_refmgr),
    m_serializableProperty(sval.m_serializableProperty)
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPropertyInfo::cPropertyInfo(const cPropertyInfo& sval, PropertyBase* serializableProperty ) :
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
    m_serializableProperty( serializableProperty )
{
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
cPropertyInfo& cPropertyInfo::Kind( eKind kind )
{
    m_kind = kind;
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
cPropertyInfo& cPropertyInfo::operator=(cPropertyInfo val)
{
    m_description           = val.m_description;
    m_kind                  = val.m_kind;
    m_xmlmode               = val.m_xmlmode;
    m_enumArray	            = val.m_enumArray;
    m_register              = val.m_register;
    m_eventEnable           = val.m_eventEnable;
    m_min                   = val.m_min;
    m_max                   = val.m_max;
    m_enable                = val.m_enable;
    m_refmgr                = val.m_refmgr;
    m_serializableProperty  = val.m_serializableProperty;
    return *this;
}

}
