#pragma once
#include "Memory.h"
#include "includes.h"
#include "vector.h"

namespace gpuNN {

	class GenericMatrix {

	public:
		GenericMatrix();
		GenericMatrix(const GenericMatrix&);
		GenericMatrix(int, int, int);

		void			Release();
		GenericMatrix&  operator=(const GenericMatrix&);
		GenericMatrix&  operator<<=(GenericMatrix&);
		void			SetSize(int, int, int);
		void			Zeros();
		void			Ones();
		int				getLength()const;
		virtual void	Randu()=0;
		virtual void	Randn()=0;
		virtual void	Malloc()=0;
		virtual void	Free() = 0;
		virtual void	Memcpy(GenericMatrix& rhs) = 0;
		virtual void	SetAll(float val)=0;
		virtual void	Set(int, int, int, float)=0;
		virtual void	Set(int, int, const VectorFloat&)=0;
		virtual void	Set(int, int, float)=0;
		virtual float	Get(int, int, int)const;
		VectorFloat		Get(int, int)const;
		VectorFloat		Get(int)const;
		float*			getData();

		// TODO rethink that

		GenericMatrix& operator+(const GenericMatrix&) const;

		GenericMatrix& operator-(const GenericMatrix&) const;

	protected:
		int				m_cols;
		int				m_rows;
		int				m_channels;
		float*			m_data;
	};

	class CpuMatrix : public GenericMatrix
	{
		
		public:
			CpuMatrix();
			CpuMatrix(const GenericMatrix&);
		public:
			 void   Malloc() override;
			 void   Memcpy(GenericMatrix& rhs) override;
			 void   Free() override;
			 void	SetAll(float val) = 0;
			 void	Set(int, int, int, float) = 0;
			 void	Set(int, int, const VectorFloat&) = 0;
			 void	Set(int, int, float) = 0;
	};
	class GpuMatrix : public GenericMatrix {
		public:
			void Malloc();
			void Free();
	};
}