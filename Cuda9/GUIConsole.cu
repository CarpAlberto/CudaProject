#include "GUIConsole.h"
using namespace gpuNN;

void GUIConsole::showMessage(const std::string& message) {
	m_mutex.lock();
	std::cout << message;
	m_mutex.unlock();
}

void GUIConsole::showErrorMessage(const std::string& errorMessage) {
	m_mutex.lock();
	std::cout << "ERROR:" << errorMessage << std::endl;
	m_mutex.unlock();
}

void GUIConsole::Show(double value) {
	m_mutex.lock();
	std::cout << value;
	m_mutex.unlock();
}

void GUIConsole::Show(int value) {
	m_mutex.lock();
	std::cout << value;
	m_mutex.unlock();
}