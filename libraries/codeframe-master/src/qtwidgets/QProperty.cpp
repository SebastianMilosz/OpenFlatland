#include "QProperty.h"
#include <QSpinBox>
#include <QTextEdit>
#include <QCheckBox>
#include <QComboBox>
#include <QFile>
#include <QScrollBar>
#include <QSlider>
#include <qtserializableutilities.h>
#include <QDebug>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QProperty::slotPropertyChanged( codeframe::Property* prop )
{
    emit valueChanged( prop );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QProperty::QProperty(const QString& name /*= QString()*/, codeframe::Property* propertyObject /*= 0*/, QObject* parent /*= 0*/) : QObject(parent),
m_propertyObject(propertyObject)
{
    if( propertyObject )
    {
        propertyObject->signalChanged.connect( this, &QProperty::slotPropertyChanged );
    }

	setObjectName(name);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
codeframe::eKind QProperty::type( int role ) const
{
    (void)role;

    return m_propertyObject->Info().GetKind();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QVariant QProperty::value( int role ) const
{
	if (m_propertyObject)
    {
        if( role == Qt::ToolTipRole )
        {
            return QVariant( QString::fromStdString((std::string)( m_propertyObject->Info().GetDescription() )) );
        }
        else
        {
            switch( type() )
            {
                case codeframe::KIND_NON:          return QVariant();
                case codeframe::KIND_LOGIC:        return QVariant( (bool)(char)( *m_propertyObject ) );
                case codeframe::KIND_NUMBER:       return QVariant( (int)       ( *m_propertyObject ) );
                case codeframe::KIND_NUMBERRANGE:  return QVariant( (int)       ( *m_propertyObject ) );
                case codeframe::KIND_REAL:         return QVariant( (double)    ( *m_propertyObject ) );
                case codeframe::KIND_TEXT:         return QVariant( QString::fromStdString((std::string)(*m_propertyObject)) );
                case codeframe::KIND_ENUM:
                {
                    switch( role )
                    {
                        case Qt::DecorationRole:
                        case Qt::DisplayRole:
                            return QVariant( QString::fromStdString((std::string)( m_propertyObject->ToString() )) );
                        default:
                            return QVariant( (int)( *m_propertyObject ) );
                    }
                }
                case codeframe::KIND_DIR:          return QVariant( QString::fromStdString((std::string)(*m_propertyObject)) );
                case codeframe::KIND_URL:          return QVariant( QString::fromStdString((std::string)(*m_propertyObject)) );
                case codeframe::KIND_FILE:         return QVariant( QString::fromStdString((std::string)(*m_propertyObject)) );
                case codeframe::KIND_DATE:         return QVariant( QString::fromStdString((std::string)(*m_propertyObject)) );
                case codeframe::KIND_FONT:         return QVariant( QString::fromStdString((std::string)(*m_propertyObject)) );
                case codeframe::KIND_COLOR:        return QVariant();
                case codeframe::KIND_IMAGE:        return QVariant();
            }
        }
    }

    return QVariant();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QProperty::setValue(const QVariant &value)
{
	if (m_propertyObject)
    {
        switch( type() )
        {
            case codeframe::KIND_NON:          *m_propertyObject = 0; break;
            case codeframe::KIND_LOGIC:        *m_propertyObject = value.toBool(); break;
            case codeframe::KIND_NUMBER:       *m_propertyObject = value.toInt(); break;
            case codeframe::KIND_NUMBERRANGE:  *m_propertyObject = value.toInt(); break;
            case codeframe::KIND_REAL:         *m_propertyObject = value.toDouble(); break;
            case codeframe::KIND_TEXT:         *m_propertyObject = value.toString().toStdString(); break;
            case codeframe::KIND_ENUM:         *m_propertyObject = value.toInt(); break;
            case codeframe::KIND_DIR:          *m_propertyObject = value.toString().toStdString(); break;
            case codeframe::KIND_URL:          *m_propertyObject = value.toString().toStdString(); break;
            case codeframe::KIND_FILE:         *m_propertyObject = value.toString().toStdString(); break;
            case codeframe::KIND_DATE:         *m_propertyObject = value.toString().toStdString(); break;
            case codeframe::KIND_FONT:         *m_propertyObject = value.toString().toStdString(); break;
            case codeframe::KIND_COLOR:        *m_propertyObject = 0; break;
            case codeframe::KIND_IMAGE:        *m_propertyObject = 0; break;
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool QProperty::isReadOnly()
{
    if (m_propertyObject)
    {
        if( m_propertyObject->Info().GetEnable() )
            return false;
        else
            return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QWidget* QProperty::createEditor(QWidget *parent, const QStyleOptionViewItem& /*option*/)
{
	QWidget* editor = 0;

    switch( type() )
	{
        case codeframe::KIND_NON:
            break;
        case codeframe::KIND_LOGIC:
        {
            QComboBox* editorCb = new QComboBox( parent );
            QStringList arrDiet;
            arrDiet << "false";
            arrDiet << "true";
            editorCb->insertItems(0, arrDiet);
            connect(editorCb, SIGNAL(currentIndexChanged(int)), this, SLOT(setValue(int)));

            editor = editorCb;

            break;
        }
        case codeframe::KIND_NUMBER:
            if( m_propertyObject )
            {
                QSpinBox* seditor = new QSpinBox( parent );

                seditor->setMaximum( m_propertyObject->Info().GetMax() );
                seditor->setMinimum( m_propertyObject->Info().GetMin() );

                editor = seditor;
                connect(editor, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));
            }
            break;
        case codeframe::KIND_NUMBERRANGE:
            if( m_propertyObject )
            {
                QSlider* seditor = new QSlider( Qt::Horizontal, parent );

                seditor->setMaximum( m_propertyObject->Info().GetMax() );
                seditor->setMinimum( m_propertyObject->Info().GetMin() );

                editor = seditor;
                connect(editor, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));
            }
            break;
        case codeframe::KIND_REAL:
            if( m_propertyObject )
            {
                QDoubleSpinBox* seditor = new QDoubleSpinBox( parent );

                seditor->setMaximum( m_propertyObject->Info().GetMax() );
                seditor->setMinimum( m_propertyObject->Info().GetMin() );

                editor = seditor;
                connect(editor, SIGNAL(valueChanged(double)), this, SLOT(setValue(double)));
            }
            break;
        case codeframe::KIND_TEXT:
            editor = new QTextEdit( parent );
            connect(editor, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));
            break;
        case codeframe::KIND_ENUM:
            editor = new QComboBox( parent );
            codeframe::qtSerializableUtilities::FillQComboBox( (QComboBox*)editor , *m_propertyObject );
            connect(editor, SIGNAL(currentIndexChanged(int)), this, SLOT(setValue(int)));
            break;
        case codeframe::KIND_DIR:
            editor = new QTextEdit( parent );
            connect(editor, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));
            break;
        case codeframe::KIND_URL:
            editor = new QTextEdit( parent );
            connect(editor, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));
            break;
        case codeframe::KIND_FILE:
            editor = new QTextEdit( parent );
            connect(editor, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));
            break;
        case codeframe::KIND_DATE:
            break;
        case codeframe::KIND_FONT:
            break;
        case codeframe::KIND_COLOR:
            break;
        case codeframe::KIND_IMAGE:
            break;
        default:
            return editor;
	}
	return editor;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool QProperty::setEditorData(QWidget *editor, const QVariant &data)
{
    switch( type() )
	{
        case codeframe::KIND_NON:
            return false;
        case codeframe::KIND_LOGIC:
            editor->blockSignals(true);
            static_cast<QComboBox*>(editor)->setCurrentIndex( data.toBool() );
            editor->blockSignals(false);
            return true;
        case codeframe::KIND_NUMBER:
            editor->blockSignals(true);
            static_cast<QSpinBox*>(editor)->setValue( data.toInt() );
            editor->blockSignals(false);
            return true;
        case codeframe::KIND_NUMBERRANGE:
            editor->blockSignals(true);
            static_cast<QSlider*>(editor)->setValue( data.toInt() );
            editor->blockSignals(false);
            return true;
        case codeframe::KIND_REAL:
            editor->blockSignals(true);
            static_cast<QDoubleSpinBox*>(editor)->setValue( data.toDouble() );
            editor->blockSignals(false);
            return true;
        case codeframe::KIND_TEXT:
            return false;
        case codeframe::KIND_ENUM:
            editor->blockSignals(true);
            static_cast<QComboBox*>(editor)->setCurrentIndex( data.toInt() );
            editor->blockSignals(false);
            return true;
        case codeframe::KIND_DIR:
            return false;
        case codeframe::KIND_URL:
            return false;
        case codeframe::KIND_FILE:
            return false;
        case codeframe::KIND_DATE:
            return false;
        case codeframe::KIND_FONT:
            return false;
        case codeframe::KIND_COLOR:
            return false;
        case codeframe::KIND_IMAGE:
            return false;
        default:
            return false;
	}
	return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QVariant QProperty::editorData(QWidget *editor)
{
    switch( type() )
	{
        case codeframe::KIND_NON:
            break;
        case codeframe::KIND_LOGIC:
            return QVariant( qobject_cast<QComboBox*>(editor)->currentText() ); // currentIndex()
        case codeframe::KIND_NUMBER:
            return QVariant( qobject_cast<QSpinBox*>(editor)->value() );
        case codeframe::KIND_NUMBERRANGE:
            return QVariant( qobject_cast<QSlider*>(editor)->value() );
        case codeframe::KIND_REAL:
            return QVariant(qobject_cast<QDoubleSpinBox*>(editor)->value());
        case codeframe::KIND_TEXT:
            return QVariant( qobject_cast<QTextEdit*>(editor)->toPlainText() );
        case codeframe::KIND_ENUM:
            return QVariant( qobject_cast<QComboBox*>(editor)->currentIndex() );
        case codeframe::KIND_DIR:
            return QVariant( qobject_cast<QTextEdit*>(editor)->toPlainText() );
        case codeframe::KIND_URL:
            return QVariant( qobject_cast<QTextEdit*>(editor)->toPlainText() );
        case codeframe::KIND_FILE:
            return QVariant( qobject_cast<QTextEdit*>(editor)->toPlainText() );
        case codeframe::KIND_DATE:
            break;
        case codeframe::KIND_FONT:
            break;
        case codeframe::KIND_COLOR:
            break;
        case codeframe::KIND_IMAGE:
            break;
        default:
        {
            return QVariant();
        }
	}

    return QVariant();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
codeframe::Property* QProperty::findPropertyObject( QObject* propertyObject )
{
    (void)propertyObject;

	return 0;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QProperty::setValue(bool value)
{
    setValue( QVariant(value) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QProperty::setValue(double value)
{
    setValue( QVariant(value) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QProperty::setValue(int value)
{
    setValue( QVariant(value) );
}
