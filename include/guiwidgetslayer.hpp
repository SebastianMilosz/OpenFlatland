#ifndef GUIWIDGETSLAYER_HPP
#define GUIWIDGETSLAYER_HPP

#include <map>

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
                GuiDataStorage( const std::string& name ) :
                    m_name( name )
                {
                    s_InstanceHandlerVector.push_back( this );
                }

                virtual ~GuiDataStorage()
                {
                    // @todo Remove handlers on destruct

                }

                static void Init( ImGuiContext* context )
                {
                    if ( NULL != context )
                    {
                        // Add .ini handle for persistent docking data
                        ImGuiSettingsHandler ini_handler;
                        ini_handler.TypeName = "WidgetsData";
                        ini_handler.TypeHash = ImHashStr("WidgetsData", 0, 0);
                        ini_handler.ReadOpenFn = &GuiDataStorage::GuiSettingsHandler_ReadOpen;
                        ini_handler.ReadLineFn = &GuiDataStorage::GuiSettingsHandler_ReadLine;
                        ini_handler.WriteAllFn = &GuiDataStorage::GuiSettingsHandler_WriteAll;
                        context->SettingsHandlers.push_back(ini_handler);
                    }
                }

                virtual void Add( const std::string& key, const std::string& value )
                {
                    m_DataMap[ key ] = value;
                }

                virtual void Get( const std::string& key, std::string& value )
                {

                }
            private:
                static void* GuiSettingsHandler_ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char* name)
                {
                    if (strcmp(name, "WidgetsData") != 0)
                    {
                        return NULL;
                    }
                    return (void*)1;
                }

                static void  GuiSettingsHandler_ReadLine(ImGuiContext*, ImGuiSettingsHandler*, void* entry, const char* line)
                {

                }

                static void  GuiSettingsHandler_WriteAll(ImGuiContext* imgui_ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
                {
                    std::vector<GuiDataStorage*>::iterator vit;
                    for ( vit = s_InstanceHandlerVector.begin(); vit != s_InstanceHandlerVector.end(); ++vit )
                    {
                        GuiDataStorage* datStore = *vit;
                        if ( (GuiDataStorage*)NULL != datStore )
                        {
                            std::map<std::string, std::string>::iterator it;

                            buf->appendf("[%s][Data-%s]\n", handler->TypeName, datStore->m_name.c_str());
                            for ( it = datStore->m_DataMap.begin(); it != datStore->m_DataMap.end(); it++ )
                            {
                                buf->appendf(" %s=%s", it->first.c_str(), it->second.c_str() );
                                buf->appendf("\n");
                            }

                            buf->appendf("\n");
                        }
                    }
                }

                static std::vector<GuiDataStorage*> s_InstanceHandlerVector;
                std::string m_name;
                std::map<std::string, std::string>  m_DataMap;
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

        GuiDataStorage m_GuiConsoleDataStorage;
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
