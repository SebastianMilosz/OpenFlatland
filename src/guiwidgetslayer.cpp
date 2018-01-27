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
    menu->addMenuItem("Copy");
    menu->addMenuItem("Paste");
    menu->addMenu("Help");
    menu->addMenuItem("About");
    m_gui.add( menu );

    auto button = tgui::Button::create();
    //button->setRenderer(theme.getRenderer("Button"));
    button->setPosition(75, 70);
    button->setText("OK");
    button->setSize(100, 30);
    button->connect("pressed", [=](){  });
    m_gui.add(button);


    auto child = tgui::ChildWindow::create();
    //child->setRenderer(theme.getRenderer("ChildWindow"));
    child->setSize(250, 120);
    child->setPosition(420, 80);
    child->setTitle("Child window");
    m_gui.add(child);
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
