#ifndef DRAWABLE_ENTITY_VISION_H
#define DRAWABLE_ENTITY_VISION_H

#include "entity_vision.hpp"

class DrawableEntityVision : public EntityVision
{
    public:
                 DrawableEntityVision( codeframe::ObjectNode* parent );
                 DrawableEntityVision( const DrawableEntityVision& other );
        virtual ~DrawableEntityVision();

        void setPosition(float x, float y);
        void setRotation(float angle);

        void draw( sf::RenderTarget& target, sf::RenderStates states ) const;

#ifdef ENTITY_VISION_DEBUG
        void AddDirectionRay( EntityVision::sRay ray );
#endif // ENTITY_VISION_DEBUG

        void EndFrame();
    protected:

    private:
        void PrepareRays();

        sf::ColorizeCircleShape m_visionShape;

        std::vector<sf::Vertex> m_rayLines;

#ifdef ENTITY_VISION_DEBUG
        sf::Vertex m_directionRayLine[2];
        EntityVision::sRay m_directionRay;
#endif // ENTITY_VISION_DEBUG
};

#endif // DRAWABLE_ENTITY_VISION_H
