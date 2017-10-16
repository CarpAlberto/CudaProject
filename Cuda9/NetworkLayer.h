#pragma once
class GenericMatrix;
class Vector;
#include "includes.h"

namespace gpuNN {
	
	typedef std::vector<std::unique_ptr<GenericMatrix>> Layer;

	class NetworkLayer {
	
	public:
		/// <summary>
		/// Parameter that specify if the cpu or gpu will be used for operation
		/// </summary>
		/// <param name="mode"></param>
		NetworkLayer(Bridge mode);
		virtual ~NetworkLayer();

	protected:
		std::unique_ptr<GenericMatrix> m_output_matrix;
		std::unique_ptr<GenericMatrix> m_delta_matrix;
		std::vector<Layer>			   m_output_layer;
		Bridge						   m_mode;
		std::string                    m_layer_name;

	};
}