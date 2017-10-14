#pragma once
#include "includes.h"
#include "LoggingPolicy.h"

namespace gpuNN {

	template< typename log_policy>
	class ApplicationLogger
	{
	protected:
		unsigned int log_line_number;
		std::stringstream logStream;
		log_policy* policy;
		std::mutex mutex;
		SeverityType loggingLevel;
		std::vector< std::string > log_buffer;

		void print_impl(std::stringstream&&);

		template<typename First, typename...Rest>
		void print_impl(std::stringstream&&, First&& parm1, Rest&&...parm);
	public:
		ApplicationLogger() {}
		ApplicationLogger(const std::string& name, SeverityType severity = SeverityType::DEBUG);
		std::string getTime();
		std::string getLoglineHeader();
		template< SeverityType severity, typename...Args >
		void print(Args...args);
		~ApplicationLogger();
	};


	template<typename log_policy>
	ApplicationLogger<log_policy>::ApplicationLogger(const std::string& name, SeverityType severity)
		:log_line_number(0),
		loggingLevel(severity)
	{
		this->policy->open_out_stream(name);
	}


	template< typename log_policy>
	void ApplicationLogger< log_policy >::print_impl(std::stringstream&& log_stream)
	{
		log_buffer.push_back(log_stream.str());
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
		header << log_line_number++ << " < " << get_time() << " - ";

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
		log_stream << log_line_number++ << " " << getTime() << " ";
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