#include "QVariantDelegate.h"

#include "QProperty.h"

#include <QAbstractItemView>
#include <QtCore/QSignalMapper>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QVariantDelegate::QVariantDelegate(QObject* parent) : QItemDelegate(parent)
{
	m_finishedMapper = new QSignalMapper(this);
	connect(m_finishedMapper, SIGNAL(mapped(QWidget*)), this, SIGNAL(commitData(QWidget*)));
	connect(m_finishedMapper, SIGNAL(mapped(QWidget*)), this, SIGNAL(closeEditor(QWidget*)));
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QVariantDelegate::~QVariantDelegate()
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QWidget *QVariantDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem& option , const QModelIndex & index ) const
{
	QWidget* editor = 0;
    QProperty* p = static_cast<QProperty*>(index.internalPointer());
	switch(p->value().type())
	{
    case QVariant::List:
	case QVariant::Color:
    case QVariant::Bool:
	case QVariant::Int:
	case QVariant::Double:	
	case QVariant::UserType:			
		editor = p->createEditor(parent, option);
		if (editor)	
		{
			if (editor->metaObject()->indexOfSignal("editFinished()") != -1)
			{
				connect(editor, SIGNAL(editFinished()), m_finishedMapper, SLOT(map()));
				m_finishedMapper->setMapping(editor, editor);
			}
			break; // if no editor could be created take default case
		}
	default:
		editor = QItemDelegate::createEditor(parent, option, index);
	}
	parseEditorHints(editor, p->editorHints());
	return editor;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QVariantDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{		
	m_finishedMapper->blockSignals(true);
	QVariant data = index.model()->data(index, Qt::EditRole);	
	
	switch(data.type())
	{
    case QVariant::List:
    case QVariant::Color:
    case QVariant::Bool:
    case QVariant::Double:
	case QVariant::UserType:
	case QVariant::Int:
        if (static_cast<QProperty*>(index.internalPointer())->setEditorData(editor, data)) // if editor couldn't be recognized use default
			break; 
	default:
		QItemDelegate::setEditorData(editor, index);
		break;
	}
	m_finishedMapper->blockSignals(false);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QVariantDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{	
	QVariant data = index.model()->data(index, Qt::EditRole);	
	switch(data.type())
	{
    case QVariant::List:
    case QVariant::Color:
    case QVariant::Bool:
    case QVariant::Double:
	case QVariant::UserType: 
	case QVariant::Int:
		{
            QVariant data = static_cast<QProperty*>(index.internalPointer())->editorData(editor);
			if (data.isValid())
			{
				model->setData(index, data , Qt::EditRole); 
				break;
			}
		}
	default:
		QItemDelegate::setModelData(editor, model, index);
		break;
	}
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QVariantDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex& index ) const
{
	return QItemDelegate::updateEditorGeometry(editor, option, index);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QVariantDelegate::parseEditorHints(QWidget* editor, const QString& editorHints) const
{
	if (editor && !editorHints.isEmpty())
	{
		editor->blockSignals(true);
		// Parse for property values
		QRegExp rx("(.*)(=\\s*)(.*)(;{1})");
		rx.setMinimal(true);
		int pos = 0;
		while ((pos = rx.indexIn(editorHints, pos)) != -1) 
		{
			//qDebug("Setting %s to %s", qPrintable(rx.cap(1)), qPrintable(rx.cap(3)));
			editor->setProperty(qPrintable(rx.cap(1).trimmed()), rx.cap(3).trimmed());				
			pos += rx.matchedLength();
		}
		editor->blockSignals(false);
	}
}
