#ifndef DRAWABLE_ENTITY_VISION_H
#define DRAWABLE_ENTITY_VISION_H

#include "entity_vision.hpp"
#include "drawable_object.hpp"

class DrawableEntityVision : public EntityVision, public DrawableObject
{
    public:
                 DrawableEntityVision( codeframe::ObjectNode* parent );
                 DrawableEntityVision( const DrawableEntityVision& other );
        virtual ~DrawableEntityVision();

        codeframe::Property<unsigned int> ColorizeMode;

        void setPosition(float x, float y) override;
        void setRotation(float angle) override;
        void draw( sf::RenderTarget& target, sf::RenderStates states ) const override;

    protected:
        void EndFrame() override;

#ifdef ENTITY_VISION_DEBUG
        void AddDirectionRay( EntityVision::Ray ray ) override;
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
        EntityVision::Ray m_directionRay;
#endif // ENTITY_VISION_DEBUG
};

#endif // DRAWABLE_ENTITY_VISION_H
