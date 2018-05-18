#include "Utils.h"
#include <iostream>
#include <fstream>
#include <Windows.h>
using namespace gpuNN;
using namespace std;

std::size_t Utils::CalculatePadding(const std::size_t baseAddress, const std::size_t alignment) {
	const std::size_t multiplier = (baseAddress / alignment) + 1;
	const std::size_t alignedAddress = multiplier * alignment;
	const std::size_t padding = alignedAddress - baseAddress;
	return padding;
}

std::size_t Utils::CalculatePaddingWithHeader(const std::size_t baseAddress, const std::size_t alignment, const std::size_t headerSize) {

	std::size_t padding = CalculatePadding(baseAddress, alignment);
	std::size_t neededSpace = headerSize;
	if (padding < neededSpace) {
		neededSpace -= padding;
		if (neededSpace % alignment > 0) {
			padding += alignment * (1 + (neededSpace / alignment));
		}
		else
		{
			padding += alignment * (neededSpace / alignment);
		}
	}
	return padding;
}

std::string Utils::Trim(std::string const& source, char const* delims) {
	std::string result(source);
	std::string::size_type index = result.find_last_not_of(delims);
	if (index != std::string::npos)
		result.erase(++index);

	index = result.find_first_not_of(delims);
	if (index != std::string::npos)
		result.erase(0, index);
	else
		result.erase();
	return result;

}

bool Utils::isAscii(char ch)
{
	return std::isalpha(static_cast<unsigned char>(ch));
}

std::vector<std::string> Utils::Split(std::string s, char delim)
{
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);
	while (std::getline(tokenStream, token, delim))
	{
		tokens.push_back(token);
	}
	return tokens;
}
void Utils::ReadDirectory(const std::string directory, std::vector<std::string>& array)
{
	std::string pattern(directory);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			array.push_back(data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}

}

size_t Utils::getNumberLines(std::ifstream& stream)
{
	string line;
	size_t i = 0;
	while (std::getline(stream,line)) {
		i++;
	}
	stream.clear();
	stream.seekg(0, ios::beg);
	return i;
}