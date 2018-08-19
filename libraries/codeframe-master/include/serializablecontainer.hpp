#ifndef CSERIALIZABLECONTAINER_H
#define CSERIALIZABLECONTAINER_H

#include <exception>
#include <stdexcept>
#include <vector>
#include <cstdbool>

#include <MathUtilities.h>
#include <smartpointer.h>

#include "serializable.hpp"
#include "propertyignorelist.hpp"

#define MAXID 100

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    class cSerializableContainer : public cSerializable
    {
        friend class cSerializableSelectable;

        public:
            std::string Role()      const { return "Container"; }
            std::string Class()     const { return "cSerializableContainer"; }
            std::string BuildType() const { return "Static"; }

        public:
                     cSerializableContainer( std::string name, cSerializableInterface* parentObject );
            virtual ~cSerializableContainer();

            virtual smart_ptr<cSerializableInterface> Create(
                                                             const std::string className,
                                                             const std::string objName,
                                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                             ) = 0;

            smart_ptr<cSerializableInterface> operator[]( int i );

            virtual void CreateRange( std::string className, std::string objName, int range );
            virtual bool Dispose( unsigned int id );
            virtual bool Dispose( std::string objName );
            virtual bool Dispose( smart_ptr<cSerializableInterface> obj );
            virtual bool DisposeByBuildType( std::string serType, cIgnoreList ignore = cIgnoreList() );
            virtual bool Dispose();

            int         Count() const;
            bool        IsName( std::string& name );
            std::string CreateUniqueName( std::string nameBase );
            bool        IsInRange( unsigned int cnt ) const;
            bool        Select( int pos );
            bool        IsSelected();

            smart_ptr<cSerializableInterface> GetSelected();
            smart_ptr<cSerializableInterface> Get( int id );

            int Add( smart_ptr<cSerializable> classType, int pos = -1 );

            signal1< smart_ptr<cSerializableInterface> > signalContainerSelectionChanged;

        protected:
            virtual int InsertObject( smart_ptr<cSerializable> classType, int pos = -1 );

        private:
            void slotSelectionChanged( smart_ptr<cSerializableInterface> obj );

            std::vector< smart_ptr<cSerializable> > m_containerVector;
            smart_ptr<cSerializableInterface> m_selected;

            unsigned int m_size;
    };

}

#endif // CSERIALIZABLECONTAINER_H
