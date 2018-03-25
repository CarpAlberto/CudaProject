#pragma once
#include <string>
#include <fstream>
#include <memory>

namespace gpuNN {

	/// <summary>
	/// Interface for logging 
	/// </summary>
	class LoggingInterfacePolicy
	{
	public:
		/// <summary>
		/// Open a given stream
		/// </summary>
		/// <param name="name">The name of the string</param>
		virtual void	open_ostream(const std::string& name) = 0;
		/// <summary>
		/// Closes the stream
		/// </summary>
		virtual void	close_ostream() = 0;
		/// <summary>
		/// Write a given data to the stream
		/// </summary>
		/// <param name="msg"></param>
		virtual void	write(const std::string& msg) = 0;

	};

	/// <summary>
	/// A file logging implementation for the interface
	/// </summary>
	class FileLoggingPolicy : public LoggingInterfacePolicy
	{
		std::unique_ptr< std::ofstream > out_stream;
	public:
		FileLoggingPolicy() : out_stream(new std::ofstream) 
		{

		}
		void open_ostream(const std::string& name) override;
		void close_ostream() override;
		void write(const std::string& msg) override;
		~FileLoggingPolicy();
	};
}