#ifndef GUIWIDGETSLAYER_HPP
#define GUIWIDGETSLAYER_HPP

#include <SFML/Graphics.hpp>

#include "consolewidget.hpp"
#include "propertyeditorwidget.hpp"
#include "annviewerwidget.hpp"
#include "informationwidget.hpp"

#include "imgui_internal.h"

class GUIWidgetsLayer
{
    public:
        enum { MOUSE_MODE_SEL_ENTITY, MOUSE_MODE_DEL_ENTITY, MOUSE_MODE_ADD_ENTITY, MOUSE_MODE_ADD_LINE };

    public:
        class GuiDataStorage : public utilities::data::DataStorage
        {
            public:
                GuiDataStorage()
                {
                }

                static void Init( ImGuiContext* context )
                {
                    if ( NULL != context )
                    {
                        // Add .ini handle for persistent docking data
                        ImGuiSettingsHandler ini_handler;
                        ini_handler.TypeName = "Application";
                        ini_handler.TypeHash = ImHashStr("Application", 0, 0);
                        ini_handler.ReadOpenFn = &GuiDataStorage::ApplicationSettingsHandler_ReadOpen;
                        ini_handler.ReadLineFn = &GuiDataStorage::ApplicationSettingsHandler_ReadLine;
                        ini_handler.WriteAllFn = &GuiDataStorage::ApplicationSettingsHandler_WriteAll;
                        context->SettingsHandlers.push_back(ini_handler);
                    }
                }

                virtual void Add( const std::string& key, const std::string& value )
                {

                }

                virtual void Get( const std::string& key, std::string& value )
                {

                }
            private:
                static void* ApplicationSettingsHandler_ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char* name)
                {
                    if (strcmp(name, "Application") != 0)
                    {
                        return NULL;
                    }
                    return (void*)1;
                }

                static void  ApplicationSettingsHandler_ReadLine(ImGuiContext*, ImGuiSettingsHandler*, void* entry, const char* line)
                {

                }

                static void  ApplicationSettingsHandler_WriteAll(ImGuiContext* imgui_ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
                {

                }
        };

        GUIWidgetsLayer( sf::RenderWindow& window, cSerializableInterface& parent, const std::string& configFile );
        virtual ~GUIWidgetsLayer();

        void AddGuiRegion( int x, int y, int w, int h );

        bool MouseOnGui();
        bool HandleEvent( sf::Event& event );
        void Draw();

        void        SetMouseModeString( std::string mode );
        std::string GetMouseModeString();

        void        SetMouseModeId( int mode );
        int         GetMouseModeId();

        ConsoleWidget&          GetConsoleWidget() { return m_ConsoleWidget; }
        PropertyEditorWidget&   GetPropertyEditorWidget() { return m_PropertyEditorWidget; }
        AnnViewerWidget&        GetAnnViewerWidget() { return m_AnnViewerWidget; }
        InformationWidget&      GetInformationWidget() { return m_InformationWidget; }

    protected:

    private:
        sf::RenderWindow& m_window;
        sf::Clock         m_deltaClock;

        GuiDataStorage m_GuiDataStorage;
        int            m_MouseMode;
        bool           m_mouseCapturedByGui;
        bool           m_ConsoleWidgetOpen;
        bool           m_PropertyEditorOpen;
        bool           m_AnnViewerWidgetOpen;
        bool           m_InformationWidgetOpen;

        std::vector< sf::Rect<int> > m_guiRegions;

        // Okienka Gui
        ConsoleWidget        m_ConsoleWidget;
        PropertyEditorWidget m_PropertyEditorWidget;
        AnnViewerWidget      m_AnnViewerWidget;
        InformationWidget    m_InformationWidget;
};

#endif // GUIWIDGETSLAYER_HPP
