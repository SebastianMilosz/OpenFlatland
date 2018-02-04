#include "guiwidgetslayer.h"

#include <imgui.h>
#include <imgui-SFML.h>

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/CircleShape.hpp>

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
    m_mode( 0 )
{
    ImGui::SFML::Init( m_window );
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

    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool GUIWidgetsLayer::HandleEvent( sf::Event& event )
{
    ImGui::SFML::ProcessEvent( event );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void GUIWidgetsLayer::Draw()
{
    ImGui::SFML::Update( m_window, m_deltaClock.restart() );

    ImGui::Begin("Hello, world!");
    ImGui::Button("Look at this pretty button");
    ImGui::End();

    ImGui::SFML::Render( m_window );
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
