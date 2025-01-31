#ifndef GUIWIDGETSLAYER_HPP
#define GUIWIDGETSLAYER_HPP

#include <map>

#include <SFML/Graphics.hpp>

#include "console_widget.hpp"
#include "property_editor_widget.hpp"
#include "ann_viewer_widget.hpp"
#include "entity_vision_viewer_widget.hpp"
#include "information_widget.hpp"

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
                    std::map<std::string, std::string>::iterator it = m_DataMap.find( key );
                    if ( it != m_DataMap.end() )
                    {
                        value = it->second;
                    }
                }
            private:
                static void* GuiSettingsHandler_ReadOpen(ImGuiContext*, ImGuiSettingsHandler* handler, const char* name)
                {
                    std::vector<GuiDataStorage*>::iterator vit;
                    for ( vit = s_InstanceHandlerVector.begin(); vit != s_InstanceHandlerVector.end(); ++vit )
                    {
                        GuiDataStorage* datStore = *vit;

                        if ( (GuiDataStorage*)NULL != datStore )
                        {
                            std::string dataName = std::string("Data-") + datStore->m_name;
                            if (strcmp( name, dataName.c_str() ) != 0)
                            {
                                return NULL;
                            }
                            return (void*)1;
                        }
                    }
                    return NULL;
                }

                static void  GuiSettingsHandler_ReadLine(ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line)
                {
                    std::vector<GuiDataStorage*>::iterator vit;
                    for ( vit = s_InstanceHandlerVector.begin(); vit != s_InstanceHandlerVector.end(); ++vit )
                    {
                        GuiDataStorage* datStore = *vit;

                        if ( (GuiDataStorage*)NULL != datStore )
                        {
                            line = ImStrSkipBlank(line);

                            std::string lineString(line);

                            std::size_t found = lineString.find_first_of("=");

                            if ( found != std::string::npos )
                            {
                                std::string key   = lineString.substr(0, found );
                                std::string value = lineString.substr( found+1U );
                                datStore->m_DataMap[ key ] = value;
                            }

                        }
                    }
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

        GUIWidgetsLayer( sf::RenderWindow& window, ObjectNode& parent, const std::string& configFile );
        virtual ~GUIWidgetsLayer();

        void AddGuiRegion( int x, int y, int w, int h );

        bool MouseOnGui();
        void HandleEvent(const std::optional<sf::Event>& event);
        void Draw();

        void        SetMouseModeString( std::string mode );
        std::string GetMouseModeString();

        void        SetMouseModeId( int mode );
        int         GetMouseModeId();

        ConsoleWidget&          GetConsoleWidget() { return m_ConsoleWidget; }
        PropertyEditorWidget&   GetPropertyEditorWidget() { return m_PropertyEditorWidget; }
        AnnViewerWidget&        GetAnnViewerWidget() { return m_AnnViewerWidget; }
        VisionViewerWidget&     GetVisionViewerWidget() { return m_VisionViewerWidget; }
        InformationWidget&      GetInformationWidget() { return m_InformationWidget; }

    protected:

    private:
        sf::RenderWindow& m_window;
        sf::Clock         m_deltaClock;

        GuiDataStorage m_GuiConsoleDataStorage;
        int            m_MouseMode = 0;
        bool           m_mouseCapturedByGui = false;
        bool           m_ConsoleWidgetOpen = false;
        bool           m_PropertyEditorOpen = false;
        bool           m_AnnViewerWidgetOpen = false;
        bool           m_VisionViewerWidgetOpen = false;
        bool           m_InformationWidgetOpen = false;

        std::vector< sf::Rect<int> > m_guiRegions;

        // Okienka Gui
        ConsoleWidget        m_ConsoleWidget;
        PropertyEditorWidget m_PropertyEditorWidget;
        AnnViewerWidget      m_AnnViewerWidget;
        VisionViewerWidget   m_VisionViewerWidget;
        InformationWidget    m_InformationWidget;
};

#endif // GUIWIDGETSLAYER_HPP
