#pragma once
#include "UI.h"

namespace gpuNN {
	/// <summary>
	/// Interface for any printable onject
	/// </summary>
	class IPrintableObject {

		/// <summary>
		/// Prints the object to the generic interface
		/// </summary>
		virtual void Print(UIInterface*) const = 0;
	};
}