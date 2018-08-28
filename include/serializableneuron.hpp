#ifndef SERIALIZABLENEURON_HPP_INCLUDED
#define SERIALIZABLENEURON_HPP_INCLUDED

#include <serializable.hpp>

class SerializableNeuron : public codeframe::cSerializable
{
    public:
        std::string Role()            const { return "Object";  }
        std::string Class()           const { return "Neuron";  }
        std::string BuildType()       const { return "Dynamic"; }
        std::string ConstructPatern() const { return "";        }

    public:
        SerializableNeuron( std::string name, cSerializableInterface* parent );
        virtual ~SerializableNeuron();

    protected:
        smart_ptr<codeframe::cSerializableInterface> Create(
                                                             const std::string className,
                                                             const std::string objName,
                                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                            );
};

#endif // SERIALIZABLENEURON_HPP_INCLUDED
