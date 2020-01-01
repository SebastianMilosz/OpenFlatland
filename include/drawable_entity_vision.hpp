#ifndef DRAWABLE_ENTITY_VISION_H
#define DRAWABLE_ENTITY_VISION_H

#include "entity_vision.hpp"

class DrawableEntityVision : public EntityVision, public sf::Drawable
{
    public:
                 DrawableEntityVision( codeframe::ObjectNode* parent );
                 DrawableEntityVision( const DrawableEntityVision& other );
        virtual ~DrawableEntityVision();

        void setPosition(float x, float y) override;
        void setRotation(float angle) override;
        void draw( sf::RenderTarget& target, sf::RenderStates states ) const;

    protected:
        void EndFrame() override;

#ifdef ENTITY_VISION_DEBUG
        void AddDirectionRay( EntityVision::sRay ray ) override;
#endif // ENTITY_VISION_DEBUG

        void SetRaysStartingAngle( const int value ) override;
        void SetRaysEndingAngle( const int value ) override;

    private:
        void PrepareRays();
        void SetRaysCnt( const unsigned int cnt ) override;

        sf::ColorizeCircleShape m_visionShape;

        std::vector<sf::Vertex> m_rayLines;

#ifdef ENTITY_VISION_DEBUG
        sf::Vertex m_directionRayLine[2];
        EntityVision::sRay m_directionRay;
#endif // ENTITY_VISION_DEBUG
};

#endif // DRAWABLE_ENTITY_VISION_H
