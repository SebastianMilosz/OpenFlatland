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
        SerializableNeuron(
                            std::string name,
                            cSerializableInterface* parent,
                            unsigned int inputCnt
                          );
        virtual ~SerializableNeuron();

        codeframe::Property< unsigned int >             InputsCnt;
        codeframe::Property< codeframe::Vector<float> > InputsWeights;
        codeframe::Property< float >                    Output;
};

#endif // SERIALIZABLENEURON_HPP_INCLUDED