#include "guiwidgetslayer.h"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
GUIWidgetsLayer::GUIWidgetsLayer( sf::RenderWindow& window ) :
    m_window( window ),
    m_gui( window )
{
    auto menu = tgui::MenuBar::create();
    //menu->setRenderer(theme.getRenderer("MenuBar"));
    menu->setSize((float)window.getSize().x, 22.f);
    menu->addMenu("File");
    menu->addMenuItem("Load");
    menu->addMenuItem("Save");
    menu->addMenuItem("Exit");
    menu->addMenu("Edit");
    menu->addMenuItem("Add");
    menu->addMenuItem("Select");
    menu->addMenu("Help");
    menu->addMenuItem("About");

    menu->bindCallback(&sf::Window::close, ptr, tgui::MenuBar::MenuItemClicked);

    m_gui.add( menu );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
GUIWidgetsLayer::~GUIWidgetsLayer()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool GUIWidgetsLayer::MouseOnGui()
{
    int MouseX = sf::Mouse::getPosition(m_window).x;
    int MouseY = sf::Mouse::getPosition(m_window).y;

    tgui::Vector2f mousePos = tgui::Vector2f( MouseX, MouseY );

    for (const auto& widget : m_gui.getWidgets())
    {
        if ( widget->mouseOnWidget( mousePos ) == true )
        {
            return true;
        }
    }

    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool GUIWidgetsLayer::HandleEvent( sf::Event& event )
{
    // catch the resize events
    if (event.type == sf::Event::Resized)
    {
        m_gui.setView( m_window.getView() );
    }

    return m_gui.handleEvent( event );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void GUIWidgetsLayer::Draw()
{
    m_gui.draw();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void GUIWidgetsLayer::AddGuiRegion( int x, int y, int w, int h )
{
    sf::Rect<int> rec( x, y, w, h );
    m_guiRegions.push_back( rec );
}
