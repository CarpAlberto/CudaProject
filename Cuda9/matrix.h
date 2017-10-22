#pragma once
#include "vector.h"

namespace gpuNN {

	class GenericMatrix {

	public:
		GenericMatrix();
		GenericMatrix(const GenericMatrix&);
		GenericMatrix(int, int, int);
		void				   Release();
		GenericMatrix&		   operator=(const GenericMatrix&);
		GenericMatrix&		   operator<<=(GenericMatrix&);
		void				   SetSize(int, int, int);
		void				   Zeros();
		void				   Ones();
		int					   getLength()const;
		virtual void		   Randu()=0;
		virtual void		   Randn()=0;
		virtual void		   Malloc() {};
		virtual void		   Free() = 0;
		virtual void		   Memcpy(GenericMatrix& rhs) = 0;
		virtual void		   SetAll(float val)=0;
		virtual void		   SetAll(const VectorFloat& rhs) = 0;
		virtual void		   Set(int, int, int, float)=0;
		virtual void		   Set(int, int, const VectorFloat&)=0;
		virtual void		   Set(int, int, float)=0;
		virtual float		   Get(int, int, int)const;
		VectorFloat			   Get(int, int)const;
		VectorFloat			   Get(int)const;
		float*				   getData() const;
		virtual GenericMatrix& operator+(const GenericMatrix&) const = 0;
		virtual GenericMatrix& operator+(float val) const = 0;
		virtual GenericMatrix& operator+(const VectorFloat&) const = 0;
	    virtual GenericMatrix& operator-(const GenericMatrix&) const = 0;
		virtual GenericMatrix& operator-(float val) const = 0;
		virtual GenericMatrix& operator-(const VectorFloat&) const = 0;
		virtual GenericMatrix& operator*(const GenericMatrix&) const = 0;
		virtual GenericMatrix& operator*(float val) const = 0;
		virtual GenericMatrix& operator*(const VectorFloat&) const = 0;
		virtual void Clone(const GenericMatrix&) {};
		int getCols() const ;
		int getRows() const ;
		int getChannels() const;
		virtual void Print() const;
		/// <summary>
		/// Populates the matrix with some random data
		/// </summary>
		void SetRandom();

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
			CpuMatrix(int, int, int);
		public:
			 void   Malloc() override;
			 void   Memcpy(GenericMatrix& rhs) override;
			 void   Free() override;
			 virtual void	SetAll(float val);
			 virtual void	SetAll(const VectorFloat& rhs);
			 void	Set(int, int, int, float);
			 void	Set(int, int, const VectorFloat&);
			 void	Set(int, int, float);
			 virtual void	Randu() {};
			 virtual void	Randn() {};

			 virtual GenericMatrix& operator+(const GenericMatrix&) const;
			 virtual GenericMatrix& operator+(float val) const;
			 virtual GenericMatrix& operator+(const VectorFloat&) const;

			 virtual GenericMatrix& operator-(const GenericMatrix&) const;
			 virtual GenericMatrix& operator-(float val) const;
			 virtual GenericMatrix& operator-(const VectorFloat&) const ;

			 virtual GenericMatrix& operator*(const GenericMatrix&) const;
			 virtual GenericMatrix& operator*(float val) const;
			 virtual GenericMatrix& operator*(const VectorFloat&) const;
			 virtual void Clone(const GenericMatrix&) override;
	};
	class GpuMatrix : public GenericMatrix {
	public:
		void Malloc() override {};
		void Free() override {};
		virtual void Clone(const GenericMatrix&) override {};
	};
}