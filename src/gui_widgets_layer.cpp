#include "gui_widgets_layer.hpp"

#include <chrono>
#include <ctime>
#include <imgui.h>
#include <imgui-SFML.h>

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/CircleShape.hpp>

std::vector<GUIWidgetsLayer::GuiDataStorage*> GUIWidgetsLayer::GuiDataStorage::s_InstanceHandlerVector;

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
    else if ( mode == "Add Line"   ) { m_MouseMode = MOUSE_MODE_ADD_LINE; }
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
        case MOUSE_MODE_ADD_LINE:   return "Add Line";
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
GUIWidgetsLayer::GUIWidgetsLayer(sf::RenderWindow& window, ObjectNode& parent, const std::string& configFile) :
    m_window(window),
    m_GuiConsoleDataStorage("console"),
    m_MouseMode(MOUSE_MODE_SEL_ENTITY),
    m_mouseCapturedByGui(false),
    m_ConsoleWidgetOpen(true),
    m_PropertyEditorOpen(true),
    m_AnnViewerWidgetOpen(true),
    m_VisionViewerWidgetOpen(true),
    m_InformationWidgetOpen(true),
    m_ConsoleWidget(parent),
    m_InformationWidget(window)
{
    if (ImGui::SFML::Init(m_window))
    {
        GuiDataStorage::Init(ImGui::GetCurrentContext());

        // Change imgui config file
        ImGuiIO& IOS = ImGui::GetIO();
        IOS.IniFilename = configFile.c_str();

        if (IOS.IniFilename)
        {
            ImGui::LoadIniSettingsFromDisk(IOS.IniFilename);
        }

        m_ConsoleWidget.Load(m_GuiConsoleDataStorage);
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
GUIWidgetsLayer::~GUIWidgetsLayer()
{
    m_ConsoleWidget.Save(m_GuiConsoleDataStorage);
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
void GUIWidgetsLayer::HandleEvent(const std::optional<sf::Event>& event)
{
    if (event.has_value())
    {
        ImGui::SFML::ProcessEvent(m_window, event.value());
    }
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
            if (ImGui::MenuItem("Add Line"  , NULL, (m_MouseMode == MOUSE_MODE_ADD_LINE   ? true : false) )) { m_MouseMode = MOUSE_MODE_ADD_LINE;   }
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
            if (ImGui::MenuItem("Console", NULL, m_ConsoleWidgetOpen)) { if( m_ConsoleWidgetOpen ) m_ConsoleWidgetOpen = false; else m_ConsoleWidgetOpen = true; }
            if (ImGui::MenuItem("PropertyEditor", NULL, m_PropertyEditorOpen)) { if( m_PropertyEditorOpen ) m_PropertyEditorOpen = false; else m_PropertyEditorOpen = true; }
            if (ImGui::MenuItem("AnnViewer", NULL, m_AnnViewerWidgetOpen)) { if( m_AnnViewerWidgetOpen ) m_AnnViewerWidgetOpen = false; else m_AnnViewerWidgetOpen = true; }
            if (ImGui::MenuItem("Informations", NULL, m_InformationWidgetOpen)) { if( m_InformationWidgetOpen ) m_InformationWidgetOpen = false; else m_InformationWidgetOpen = true; }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (m_InformationWidgetOpen == true)
    {
        m_InformationWidget.Draw("Information", &m_InformationWidgetOpen);
    }

    if (m_ConsoleWidgetOpen == true)
    {
        m_ConsoleWidget.Draw("Console", &m_ConsoleWidgetOpen);
    }

    if (m_PropertyEditorOpen == true)
    {
        m_PropertyEditorWidget.Draw("Property Editor", &m_PropertyEditorOpen);
    }

    if (m_AnnViewerWidgetOpen == true)
    {
        m_AnnViewerWidget.Draw("Ann Viewer", &m_AnnViewerWidgetOpen);
    }

    if (m_VisionViewerWidgetOpen == true)
    {
        m_VisionViewerWidget.Draw("Vision Viewer", &m_VisionViewerWidgetOpen);
    }

    ImGui::SFML::Render(m_window);

    ImGuiIO& IOS = ImGui::GetIO();

    m_mouseCapturedByGui = IOS.WantCaptureMouse;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void GUIWidgetsLayer::AddGuiRegion(int x, int y, int w, int h)
{
    sf::Rect<int> rec({x, y}, {w, h});
    m_guiRegions.push_back(rec);
}
