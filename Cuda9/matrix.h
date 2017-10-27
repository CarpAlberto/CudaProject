#pragma once
#include "vector.h"

namespace gpuNN {

	/// <summary>
	/// Generic Matrix 
	/// </summary>
	class GenericMatrix : public IPrintableObject {

	public:
		/// <summary>
		/// Default Constructor for Generic Matrix
		/// </summary>
		GenericMatrix();
		/// <summary>
		/// Copy Constructor
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		GenericMatrix(const GenericMatrix& rhs);
		/// <summary>
		/// Construct the Generic matrix based on the number of rows
		/// columns and channels
		/// </summary>
		/// <param name="rows">The number of rows</param>
		/// <param name="cols">The number of column</param>
		/// <param name="channels">The number of channels</param>
		GenericMatrix(int rows, int cols, int channels);
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
		/// <summary>
		/// Returns the element from position <code>Position</code>
		/// </summary>
		/// <param name="position">The position</param>
		/// <returns>A Vecor Float Object</returns>
		VectorFloat			   Get(int position)const;
		/// <summary>
		/// Returns the Raw Data
		/// </summary>
		/// <returns></returns>
		float*				   getData() const;

		/// <summary>
		/// Performs the addition between the object and the <code>rhs</code>
		/// parameters.The caller MUST deallocate the memory.
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>The result of the addition</returns>
		virtual GenericMatrix& operator+(const GenericMatrix&) const = 0;
		/// <summary>
		/// Performs the addition between the object and the <code>rhs</code>
		/// parameters.The caller MUST deallocate the memory.
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>The result of the addition</returns>
		virtual GenericMatrix& operator+(float val) const = 0;
		/// <summary>
		/// Performs the addition between the object and the <code>rhs</code>
		/// parameters.The caller MUST deallocate the memory.
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>The result of the addition</returns>
		virtual GenericMatrix& operator+(const VectorFloat&) const = 0;

		/// <summary>
		/// Performs the substraction between the object and the <code>rhs</code>
		/// parameters.The caller MUST deallocate the memory.
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>The result of the substraction</returns>
	    virtual GenericMatrix& operator-(const GenericMatrix& rhs) const = 0;
		/// <summary>
		/// Performs the substraction between the object and the <code>rhs</code>
		/// parameters.The caller MUST deallocate the memory.
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>The result of the substraction</returns>
		virtual GenericMatrix& operator-(float rhs) const = 0;
		/// <summary>
		/// Performs the substraction between the object and the <code>rhs</code>
		/// parameters.The caller MUST deallocate the memory.
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>The result of the substraction</returns>
		virtual GenericMatrix& operator-(const VectorFloat& rhs) const = 0;

		/// <summary>
		/// Performs the multiplication between the object and the <code>rhs</code>
		/// parameters.The caller MUST deallocate the memory.
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>The result of the multiplication</returns>
		virtual GenericMatrix& operator*(const GenericMatrix& rhs) const = 0;
		/// <summary>
		/// Performs the multiplication between the object and the <code>rhs</code>
		/// parameters.The caller MUST deallocate the memory.
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>The result of the multiplication</returns>
		virtual GenericMatrix& operator*(float rhs) const = 0;
		
		/// <summary>
		/// Alocates and returns a reference to the Transposed variant of
		/// the matrix.The caller MUST deallocate the matrix
		/// </summary>
		/// <returns>A reference to the transposed matrix </returns>
		virtual GenericMatrix& Transpose() const = 0;
		/// <summary>
		/// Default implementation for cloning operation
		/// </summary>
		virtual void Clone(const GenericMatrix&) {};

		/// <summary>
		/// Gets the number of columns
		/// </summary>
		/// <returns>The number of columns</returns>
		int getCols() const ;
		/// <summary>
		/// Gets the number of rows
		/// </summary>
		/// <returns>The number of rows</returns>
		int getRows() const ;
		/// <summary>
		/// Returns the number of columns
		/// </summary>
		/// <returns>The number of columns</returns>
		int getChannels() const;
		/// <summary>
		/// Prints the object to the generic interface
		/// </summary>
		/// <param name="rhs">The Generic interface</param>
		virtual void Print(UIInterface* rhs) const override;
		/// <summary>
		/// Populates the matrix with some random data
		/// </summary>
		void SetRandom();
		/// <summary>
		/// The virtual destructor
		/// </summary>
		virtual ~GenericMatrix() {};
		/// <summary>
		/// Returns the elements as a matrix from std
		/// </summary>
		/// <returns>The Matrix of doubles</returns>
		virtual mDouble getAsMatrix() = 0;
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
			 virtual GenericMatrix& Transpose() const;
			 virtual ~CpuMatrix();
			 virtual mDouble getAsMatrix();
	};
	class GpuMatrix : public GenericMatrix {
	public:
		void Malloc() override {};
		void Free() override {};
		virtual void Clone(const GenericMatrix&) override {};
	};
}