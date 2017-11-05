#include "ApplicationContext.h"

using namespace gpuNN;

ApplicationContext::ApplicationContext()
{
	m_gui_interface		= std::make_shared<GUIConsole>();
	applicationLogging  = std::make_shared<FileLogging>("application.log",SeverityType::DEBUG);
	configProperties	= std::make_shared<StringConfiguration>("config.cfg");
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

std::shared_ptr<StringConfiguration> ApplicationContext::getConfiguration() const {

	return this->configProperties;
}