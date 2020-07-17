#include "serializable_selectable.hpp"

namespace codeframe
{

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSelectable::cSelectable( ObjectNode& sint ) :
    m_selected( false ),
    m_smartThis( smart_ptr<ObjectNode>( nullptr ) )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSelectable::~cSelectable()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSelectable::Select( bool state )
{
    m_selected = state;

    if ( smart_ptr_isValid( m_smartThis ) == true )
    {
        signalSelectionChanged.Emit( m_smartThis );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool cSelectable::IsSelected()
{
    return m_selected;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSelectable::DisconectFromContainer()
{
    //m_smartThis = smart_ptr<ObjectNode>( nullptr );
    signalSelectionChanged.disconnect_all();
}

}
