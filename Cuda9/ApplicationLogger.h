#pragma once
#include "includes.h"
#include "LoggingPolicy.h"

namespace gpuNN {

	/// <summary>
	/// The main logging class
	/// </summary>
	template< typename log_policy>
	class ApplicationLogger
	{
	protected:
		/// <summary>
		/// The current log line number
		/// </summary>
		unsigned int m_logLine;
		/// <summary>
		/// The logger stream
		/// </summary>
		std::stringstream logStream;
		/// <summary>
		/// The logging policy
		/// </summary>
		log_policy* policy;
		/// <summary>
		/// Internal mutex
		/// </summary>
		std::mutex mutex;
		/// <summary>
		/// The level logging
		/// </summary>
		SeverityType loggingLevel;
		/// <summary>
		/// Internal Buffer
		/// </summary>
		std::vector< std::string > m_LogBuffer;

		/// <summary>
		/// Base Print Implementation
		/// </summary>
		/// <param name="rhs">The value to be logged</param>
		void print_impl(std::stringstream&& rhs);
		/// <summary>
		/// Recursive Print Implementation
		/// </summary>
		template<typename First, typename...Rest>
		void print_impl(std::stringstream&&, First&& parm1, Rest&&...parm);

	public:
		/// <summary>
		/// Default Constructor
		/// </summary>
		ApplicationLogger() {}
		/// <summary>
		/// Construt the object based on the name and severity
		/// </summary>
		/// <param name="name">The name</param>
		/// <param name="severity">The severity</param>
		ApplicationLogger(const std::string& name, SeverityType severity = SeverityType::DEBUG);
		/// <summary>
		/// Gets the Time
		/// </summary>
		/// <returns></returns>
		std::string getTime();
		/// <summary>
		/// Gets the Header of the log
		/// </summary>
		/// <returns></returns>
		std::string getLoglineHeader();
		/// <summary>
		/// Public Print function
		/// </summary>
		template< SeverityType severity, typename...Args >
		void print(Args...args);
		/// <summary>
		/// The Destructor of the object
		/// </summary>
		~ApplicationLogger();
	};


	template<typename log_policy>
	ApplicationLogger<log_policy>::ApplicationLogger(const std::string& name, SeverityType severity)
		:m_logLine(0),
		loggingLevel(severity)
	{
		this->policy = new log_policy();
		this->policy->open_ostream(name);
	}


	template< typename log_policy>
	void ApplicationLogger< log_policy >::print_impl(std::stringstream&& log_stream)
	{
		m_LogBuffer.push_back(log_stream.str());
	}

	template<typename log_policy>
	template< typename First, typename...Rest >
	void ApplicationLogger< log_policy >::print_impl(std::stringstream&& log_stream, First&& parm1, Rest&&...parm)
	{
		log_stream << parm1;
		print_impl(std::forward<std::stringstream>(log_stream),
			std::move(parm)...);
	}


	template<typename log_policy>
	std::string ApplicationLogger< log_policy >::getTime()
	{
		std::string time_str;
		time_t raw_time;
		time(&raw_time);

		time_str = ctime(&raw_time);
		return time_str.substr(0, time_str.size() - 1);
	}

	template< typename log_policy >
	std::string ApplicationLogger< log_policy >::getLoglineHeader()
	{
		std::stringstream header;

		header.str("");
		header.fill('0');
		header.width(7);
		header << this->m_logLine++ << " < " << get_time() << " - ";

		header.fill('0');
		header.width(7);
		header << clock() << " > ~ ";

		return header.str();
	}

	template< typename log_policy >
	ApplicationLogger< log_policy >::~ApplicationLogger()
	{
		if (policy)
		{
			policy->close_ostream();
			delete policy;
		}
	}

	template<typename log_policy>
	template<SeverityType severity, typename ...Args>
	inline void ApplicationLogger<log_policy>::print(Args ...args)
	{

		if (severity < this->loggingLevel) {
			return;//Level too low
		}

		std::stringstream log_stream;
		auto now = std::chrono::system_clock::now();
		log_stream << this->m_logLine++ << " " << getTime() << " ";
		switch (severity)
		{
			case SeverityType::DEBUG:
				log_stream << "DBG: ";
				break;
			case SeverityType::WARNING:
				log_stream << "WRN: ";
				break;
			case SeverityType::ERROR:
				log_stream << "ERR: ";
				break;
		};

		print_impl(std::forward<std::stringstream>(log_stream), std::move(args)...);
	}
}