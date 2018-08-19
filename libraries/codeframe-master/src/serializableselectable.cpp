#include "serializableselectable.hpp"

namespace codeframe
{

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSerializableSelectable::cSerializableSelectable() :
    m_selected( false ),
    m_smartThis( smart_ptr<cSerializableInterface>(NULL) )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSerializableSelectable::~cSerializableSelectable()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableSelectable::Select( bool state )
{
    m_selected = state;

    if( smart_ptr_isValid( m_smartThis ) == true )
    {
        signalSelectionChanged.Emit( m_smartThis );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool cSerializableSelectable::IsSelected()
{
    return m_selected;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableSelectable::DisconectFromContainer()
{
    m_smartThis = smart_ptr<cSerializableInterface>( NULL );
    signalSelectionChanged.disconnect_all();
}

}
