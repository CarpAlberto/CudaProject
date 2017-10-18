#pragma once
#include "NetworkLayer.h"
namespace gpuNN 
{
	class InputLayer :
		public NetworkLayer
	{
	protected:
		int m_batch_size;
	public:
		InputLayer();
		~InputLayer();

		void forward();
	};
}

