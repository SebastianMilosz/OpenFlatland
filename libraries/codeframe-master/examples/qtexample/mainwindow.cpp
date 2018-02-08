#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->widget_serializablePropertyEditor->setObject( &m_objectContainer );
}

MainWindow::~MainWindow()
{
    delete ui;
}
