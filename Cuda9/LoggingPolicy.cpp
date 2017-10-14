#include "LoggingPolicy.h"

using namespace gpuNN;

void FileLoggingPolicy::open_ostream(const std::string& name)
{
	out_stream->open(name.c_str(), std::ios_base::binary | std::ios_base::out);
	if (!out_stream->is_open())
	{
		throw(std::runtime_error("LOGGER: Unable to open an output stream"));
	}
}

void FileLoggingPolicy::close_ostream()
{
	if (out_stream)
	{
		out_stream->close();
	}
}

void FileLoggingPolicy::write(const std::string& msg)
{
	(*out_stream) << msg << std::endl;
}

FileLoggingPolicy::~FileLoggingPolicy()
{
	if (out_stream)
	{
		close_ostream();
	}
}