#ifndef GUIWIDGETSLAYER_HPP
#define GUIWIDGETSLAYER_HPP

#include <SFML/Graphics.hpp>

#include "logwidget.hpp"
#include "propertyeditorwidget.hpp"
#include "annviewerwidget.hpp"

class GUIWidgetsLayer
{
    public:
        enum { MOUSE_MODE_SEL_ENTITY, MOUSE_MODE_DEL_ENTITY, MOUSE_MODE_ADD_ENTITY, MOUSE_MODE_ADD_LINE };

    public:
        GUIWidgetsLayer( sf::RenderWindow& window );
        virtual ~GUIWidgetsLayer();

        void AddGuiRegion( int x, int y, int w, int h );

        bool MouseOnGui();
        bool HandleEvent( sf::Event& event );
        void Draw();

        void        SetMouseModeString( std::string mode );
        std::string GetMouseModeString();

        void        SetMouseModeId( int mode );
        int         GetMouseModeId();

        int GetFps();

        LogWidget&              GetLogWidget() { return m_logWidget; }
        PropertyEditorWidget&   GetPropertyEditorWidget() { return m_PropertyEditorWidget; }
        AnnViewerWidget&        GetAnnViewerWidget() { return m_AnnViewerWidget; }

    protected:

    private:
        sf::RenderWindow&   m_window;
        sf::Clock           m_deltaClock;

        int                 m_MouseMode;
        bool                m_mouseCapturedByGui;
        bool                m_logWidgetOpen;
        bool                m_PropertyEditorOpen;
        bool                m_AnnViewerWidgetOpen;

        std::vector< sf::Rect<int> > m_guiRegions;

        // Okienka Gui
        LogWidget            m_logWidget;
        PropertyEditorWidget m_PropertyEditorWidget;
        AnnViewerWidget      m_AnnViewerWidget;
};

#endif // GUIWIDGETSLAYER_HPP
