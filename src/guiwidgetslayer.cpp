#include "guiwidgetslayer.h"

GUIWidgetsLayer::GUIWidgetsLayer( sf::RenderWindow& window ) :
    m_gui(window)
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
    m_gui.add(menu);

    auto button = tgui::Button::create();
    //button->setRenderer(theme.getRenderer("Button"));
    button->setPosition(75, 70);
    button->setText("OK");
    button->setSize(100, 30);
    button->connect("pressed", [=](){  });
    m_gui.add(button);
}

GUIWidgetsLayer::~GUIWidgetsLayer()
{
    //dtor
}

bool GUIWidgetsLayer::HandleEvent( sf::Event& event )
{
    return m_gui.handleEvent( event );
}

void GUIWidgetsLayer::Draw()
{
    m_gui.draw();
}
