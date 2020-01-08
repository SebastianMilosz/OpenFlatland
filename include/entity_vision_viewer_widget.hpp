#ifndef VISION_VIEWER_WIDGET_HPP
#define VISION_VIEWER_WIDGET_HPP

#include <imgui.h>
#include <imgui-SFML.h>
#include <sigslot.h>
#include <smartpointer.h>
#include <entity.hpp>

class VisionViewerWidget : public sigslot::has_slots<>
{
    public:
        VisionViewerWidget();
        virtual ~VisionViewerWidget();

        void SetObject( smart_ptr<codeframe::ObjectNode> obj );
        void Draw( const char* title, bool* p_open = nullptr );

        static constexpr unsigned int SCREEN_WIDTH = 320U;
        static constexpr unsigned int SCREEN_HEIGHT = 200U;
        static constexpr unsigned int DISTANCE_TO_SCREEN_FACTOR = 2000U;
    private:
        const sf::Color&& SetColorBrightness(const sf::Color& cl, const float bri);
        const float CalculateBrightness(const float distance );

        bool               m_lockObjectChange;
        ImVec2             m_cursorPos;
        sf::RectangleShape m_rectangle;
        sf::RenderTexture  m_displayTexture;
        smart_ptr<Entity>  m_objEntity;
};

#endif // VISION_VIEWER_WIDGET_HPP
