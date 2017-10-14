#include "ApplicationContext.h"

using namespace gpuNN;

ApplicationContext::ApplicationContext()
{
	m_gui_interface = std::make_shared<GUIConsole>();
	applicationLogging = std::make_shared<FileLogging>();
}

ApplicationContext::~ApplicationContext()
{
}

ApplicationContext* ApplicationContext::instance()
{
	static ApplicationContext * context = new ApplicationContext();
	return context;
}

void ApplicationContext::destroy()
{
	// TODO how to deleteh the instance
}

std::shared_ptr<UIInterface> ApplicationContext::getGUI() const
{
	return this->m_gui_interface;
}

std::shared_ptr<FileLogging> ApplicationContext::getLog() const
{
	return this->applicationLogging;
}