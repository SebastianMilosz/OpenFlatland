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

    private:
        sf::Image   m_displayImage;
        sf::Texture m_displayTexture;
        smart_ptr<Entity> m_objEntity;
};

#endif // VISION_VIEWER_WIDGET_HPP
