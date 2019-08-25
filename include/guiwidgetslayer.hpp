#ifndef GUIWIDGETSLAYER_HPP
#define GUIWIDGETSLAYER_HPP

#include <SFML/Graphics.hpp>

#include "consolewidget.hpp"
#include "propertyeditorwidget.hpp"
#include "annviewerwidget.hpp"
#include "informationwidget.hpp"

class GUIWidgetsLayer
{
    public:
        enum { MOUSE_MODE_SEL_ENTITY, MOUSE_MODE_DEL_ENTITY, MOUSE_MODE_ADD_ENTITY, MOUSE_MODE_ADD_LINE };

    public:
        GUIWidgetsLayer( sf::RenderWindow& window, cSerializableInterface& parent, utilities::data::DataStorage& ds );
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

        int                 m_MouseMode;
        bool                m_mouseCapturedByGui;
        bool                m_ConsoleWidgetOpen;
        bool                m_PropertyEditorOpen;
        bool                m_AnnViewerWidgetOpen;
        bool                m_InformationWidgetOpen;

        std::vector< sf::Rect<int> > m_guiRegions;

        // Okienka Gui
        ConsoleWidget        m_ConsoleWidget;
        PropertyEditorWidget m_PropertyEditorWidget;
        AnnViewerWidget      m_AnnViewerWidget;
        InformationWidget    m_InformationWidget;
};

#endif // GUIWIDGETSLAYER_HPP
