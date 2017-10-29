#include "Utils.h"

using namespace gpuNN;

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