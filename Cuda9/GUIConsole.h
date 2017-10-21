#pragma once
#include "UI.h"
#include <mutex>
#include <string>

namespace gpuNN {
	/// <summary>
	/// Console implementation for the UIInterface
	/// </summary>
	class GUIConsole : public UIInterface {

	protected:
		/// <summary>
		/// Mutex used for logging
		/// </summary>
		std::mutex m_mutex;
	public:
		/// <summary>
		/// Display an error message to the screen
		/// </summary>
		/// <param name="message">The message to be displayed</param>
		virtual void showMessage(const std::string& message);

		/// <summary>
		/// Shows an error message to the screen
		/// </summary>
		/// <param name="errorMessage">The error message</param>
		virtual void showErrorMessage(const std::string& errorMessage);

		/// <summary>
		/// Show a specific message to the console
		/// </summary>
		virtual void Show( double value) {
			std::cout << value;
		}
	};
}