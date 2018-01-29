#include "guiwidgetslayer.h"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void GUIWidgetsLayer::SetMode( int mode )
{
    m_mode = mode;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int GUIWidgetsLayer::GetMode()
{
    return m_mode;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
GUIWidgetsLayer::GUIWidgetsLayer( sf::RenderWindow& window ) :
    m_window( window ),
    m_gui( window ),
    m_mode( 0 )
{
    tgui::Button::Ptr button_add = tgui::Button::create();
    //button->setRenderer(theme.getRenderer("Button"));
    button_add->setPosition(5, 5);
    button_add->setText("<Add>");
    button_add->setSize(50, 30);
    m_gui.add(button_add);

    tgui::Button::Ptr button_sel = tgui::Button::create();
    //button->setRenderer(theme.getRenderer("Button"));
    button_sel->setPosition(5, 35);
    button_sel->setText("Sel");
    button_sel->setSize(50, 30);
    m_gui.add(button_sel);

    button_add->connect("pressed", [=](){ button_add->setText("<Add>"); button_sel->setText(" Sel "); this->SetMode( 0 ); });
    button_sel->connect("pressed", [=](){ button_add->setText(" Add "); button_sel->setText("<Sel>"); this->SetMode( 1 ); });
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
