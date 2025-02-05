#ifndef ENTITYGHOST_HPP
#define ENTITYGHOST_HPP

#include "entity_shell.hpp"
#include "artificial_neuron_engine.hpp"

class EntityGhost : public EntityShell
{
        CODEFRAME_META_CLASS_NAME( "EntityGhost" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
        EntityGhost( const std::string& name, int x, int y );
        virtual ~EntityGhost() = default;

        EntityGhost(const EntityGhost& other);
        EntityGhost& operator=(const EntityGhost& other);

        void CalculateNeuralNetworks();

        ArtificialNeuronEngine& GetEngine() { return m_NeuronEngine; }
    protected:
        ArtificialNeuronEngine m_NeuronEngine;

    private:
};

#endif // ENTITYGHOST_HPP
