#pragma once
#include "includes.h"

#include "GUIConsole.h"
#include "ApplicationLogger.h"
#include "LoggingPolicy.h"

namespace gpuNN {
	
	typedef ApplicationLogger<FileLoggingPolicy> FileLogging;

	/// <summary>
	/// The main application context
	/// </summary>
	class ApplicationContext
	{
	protected:
		/// <summary>
		/// The interface of the GUI 
		/// </summary>
		std::shared_ptr<UIInterface> m_gui_interface;

		/// <summary>
		/// The global application logging
		/// </summary>
		std::shared_ptr<FileLogging> applicationLogging;

		/// <summary>
		/// Destroy the unique instance
		/// </summary>
		static void destroy();

		/// <summary>
		/// The constructor of the class
		/// </summary>
		ApplicationContext();
		/// <summary>
		/// The destructor of the application
		/// </summary>
		~ApplicationContext();

	public:
		/// <summary>
		/// Returns the unique instance of the application context
		/// </summary>
		/// <returns>The unique instance of the application context</returns>
		static ApplicationContext* instance();

		/// <summary>
		/// Returns the GUI interface 
		/// </summary>
		/// <returns>Returns the GUI interface</returns>
		std::shared_ptr<UIInterface> getGUI() const;

		/// <summary>
		/// Returns the log from the application
		/// 
		/// </summary>
		/// <returns></returns>
		std::shared_ptr<FileLogging> getLog() const;
	};
}