#include "guiwidgetslayer.h"

#include <chrono>
#include <ctime>
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
void GUIWidgetsLayer::SetMouseModeString( std::string mode )
{
         if ( mode == "Add Entity" ) { m_MouseMode = MOUSE_MODE_ADD_ENTITY; }
    else if ( mode == "Del Entity" ) { m_MouseMode = MOUSE_MODE_DEL_ENTITY; }
    else if ( mode == "Sel Entity" ) { m_MouseMode = MOUSE_MODE_SEL_ENTITY; }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string GUIWidgetsLayer::GetMouseModeString()
{
    switch ( m_MouseMode )
    {
        case MOUSE_MODE_ADD_ENTITY: return "Add Entity";
        case MOUSE_MODE_DEL_ENTITY: return "Del Entity";
        case MOUSE_MODE_SEL_ENTITY: return "Sel Entity";
        default: return "unknown";
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void GUIWidgetsLayer::SetMouseModeId( int mode )
{
    m_MouseMode = mode;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int GUIWidgetsLayer::GetMouseModeId()
{
    return m_MouseMode;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
GUIWidgetsLayer::GUIWidgetsLayer( sf::RenderWindow& window ) :
    m_window( window ),
    m_MouseMode( MOUSE_MODE_SEL_ENTITY ),
    m_mouseCapturedByGui(false),
    m_logWidgetOpen(true)
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
    ImGui::SFML::Shutdown();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool GUIWidgetsLayer::MouseOnGui()
{
    return m_mouseCapturedByGui;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool GUIWidgetsLayer::HandleEvent( sf::Event& event )
{
    ImGui::SFML::ProcessEvent( event );
    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void GUIWidgetsLayer::Draw()
{
    ImGui::SFML::Update( m_window, m_deltaClock.restart() );

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Load")) {}
            if (ImGui::MenuItem("Save")) {}
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Mode"))
        {
            if (ImGui::MenuItem("Sel Entity", NULL, (m_MouseMode == MOUSE_MODE_SEL_ENTITY ? true : false) )) { m_MouseMode = MOUSE_MODE_SEL_ENTITY; }
            if (ImGui::MenuItem("Del Entity", NULL, (m_MouseMode == MOUSE_MODE_DEL_ENTITY ? true : false) )) { m_MouseMode = MOUSE_MODE_DEL_ENTITY; }
            if (ImGui::MenuItem("Add Entity", NULL, (m_MouseMode == MOUSE_MODE_ADD_ENTITY ? true : false) )) { m_MouseMode = MOUSE_MODE_ADD_ENTITY; }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))
        {
            if (ImGui::MenuItem("Undo", "CTRL+Z")) {}
            if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {}  // Disabled item
            ImGui::Separator();
            if (ImGui::MenuItem("Cut", "CTRL+X")) {}
            if (ImGui::MenuItem("Copy", "CTRL+C")) {}
            if (ImGui::MenuItem("Paste", "CTRL+V")) {}
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Window"))
        {
            if (ImGui::MenuItem("Log", NULL, m_logWidgetOpen)) { if( m_logWidgetOpen ) m_logWidgetOpen = false; else m_logWidgetOpen = true; }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if( m_logWidgetOpen == true )
    {
        m_logWidget.Draw( "Log", &m_logWidgetOpen );
    }

    ImGui::Begin("Application Info");

    // get the current mouse position in the window
    sf::Vector2i pixelPos = sf::Mouse::getPosition( m_window );

    // convert it to world coordinates
    sf::Vector2f worldPos = m_window.mapPixelToCoords( pixelPos );

    ImGui::Text( "FPS: %d", GetFps() );

    ImGui::Text( "Screen: (%d, %d)", pixelPos.x, pixelPos.y );

    ImGui::Text( "World: (%3.2f, %3.2f)", worldPos.x, worldPos.y );

    ImGui::Text( "World: (%3.2f, %3.2f)", worldPos.x/30.f, worldPos.y/30.f );

    ImGui::End();

    ImGui::SFML::Render( m_window );

    ImGuiIO& IOS = ImGui::GetIO();

    m_mouseCapturedByGui = IOS.WantCaptureMouse;
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

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int GUIWidgetsLayer::GetFps()
{
    using namespace std::chrono;
    static int count = 0;
    static auto last = high_resolution_clock::now();
    auto now = high_resolution_clock::now();
    static int fps = 0;

    count++;

    if( duration_cast<milliseconds>(now - last).count() > 1000 )
    {
        fps = count;
        count = 0;
        last = now;
    }

    return fps;
}
