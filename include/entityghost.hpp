#ifndef ENTITYGHOST_HPP
#define ENTITYGHOST_HPP

#include "entityshell.hpp"
#include "serializableneuronlayercontainer.hpp"

class EntityGhost : public EntityShell
{
    public:
        std::string Role()      const { return "Object";      }
        std::string Class()     const { return "EntityGhost"; }
        std::string BuildType() const { return "Dynamic";     }

    public:
        EntityGhost( std::string name, int x, int y );
        virtual ~EntityGhost();
        EntityGhost(const EntityGhost& other);
        EntityGhost& operator=(const EntityGhost& other);

        void CalculateNeuralNetworks();
    protected:
        SerializableNeuronLayerContainer m_NeuronLayerContainer;

    private:
};

#endif // ENTITYGHOST_HPP
