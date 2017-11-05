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
		GenericMatrix(int rows, int cols, int channels=1);
		/// <summary>
		/// Release the object
		/// </summary>
		void				   Release();
		/// <summary>
		/// Overriding The operator =
		/// </summary>
		/// <param name="rhs">The rhs paramater</param>
		/// <returns>The New object</returns>
		GenericMatrix&		   operator=(const GenericMatrix& rhs);
		/// <summary>
		/// Sets the parameters
		/// </summary>
		/// <param name="rows">The rows</param>
		/// <param name="columns">The columns</param>
		/// <param name="channels">The channels</param>
		void				   SetSize(int rows, int columns, int channels=1);
		/// <summary>
		/// Sets the zeros on th evalue of the matrix
		/// </summary>
		void				   Zeros();
		/// <summary>
		/// Sets the matrix with ones values
		/// </summary>
		void				   Ones();
		/// <summary>
		/// Returns the length of the matrix
		/// </summary>
		/// <returns>The length of the matrix</returns>
		int					   getLength() const;

		virtual void		   Randu()=0;
		virtual void		   Randn()=0;
		/// <summary>
		/// Perform the matrix allocation
		/// </summary>
		virtual void		   Malloc() {};
		/// <summary>
		/// Free the matrix
		/// </summary>
		virtual void		   Free() = 0;
		/// <summary>
		/// Copy the object into the <code>rhs</code> parameters
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		virtual void		   Memcpy(GenericMatrix& rhs) = 0;
		/// <summary>
		/// Sets all the data with the <code>val</code>
		/// </summary>
		/// <param name="val">The value to be setted</param>
		virtual void		   SetAll(float val)=0;
		virtual void		   SetAll(const VectorFloat& rhs) = 0;
		virtual void		   Set(int, int, int, float)=0;
		virtual void		   Set(int, int, const VectorFloat&)=0;
		virtual void		   Set(int, int, float)=0;
		/// <summary>
		/// Gets the value from the given position
		/// </summary>
		/// <param name="rows">The rows value</param>
		/// <param name="cols">The cols value</param>
		/// <param name="channels">The channels value</param>
		/// <returns>The float value</returns>
		virtual float		   Get(int rows, int cols, int channels = 0)const;
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
		virtual void SetRandom();
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
	/// <summary>
	/// Cpu Matrix Managed
	/// </summary>
	class CpuMatrix : public GenericMatrix
	{
		
		public:
			CpuMatrix();
			CpuMatrix(const GenericMatrix& rhs);
			CpuMatrix(int rows, int columns, int channel = 1);
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
	/// <summary>
	/// Gpu Managed Matrix
	/// </summary>
	class GpuMatrix : public GenericMatrix {

	public:
		#pragma region Constructors
		GpuMatrix(int rows, int columns, int channel = 1);
		GpuMatrix(const GenericMatrix& rhs);
		GpuMatrix();
#pragma endregion
	public:
		void Malloc() override;
		void Free() override;
		virtual void Clone(const GenericMatrix&) override ;
		virtual void		   Memcpy(GenericMatrix& rhs);
		virtual float		   Get(int, int, int channel = 0)const override;

		virtual GenericMatrix& operator+(const GenericMatrix&) const;
		virtual GenericMatrix& operator+(float val) const { return *new GpuMatrix(); }
		virtual GenericMatrix& operator+(const VectorFloat&) const { return *new GpuMatrix(); };

		virtual GenericMatrix& operator-(const GenericMatrix&) const { return *new GpuMatrix(); }
		virtual GenericMatrix& operator-(float val) const { return *new GpuMatrix(); }
		virtual GenericMatrix& operator-(const VectorFloat&) const { return *new GpuMatrix(); };

		virtual GenericMatrix& operator*(const GenericMatrix& rhs) const { return *new GpuMatrix(); };
		virtual GenericMatrix& operator*(float val) const { return *new GpuMatrix(); };
		virtual GenericMatrix& operator*(const VectorFloat&) { return *new GpuMatrix(); };

		virtual GenericMatrix& Transpose() const { return *new GpuMatrix(); };

		virtual void		   SetAll(float val) {};
		virtual void		   SetAll(const VectorFloat& rhs) {};
		virtual void		   Set(int, int, int, float);
		void				   Set(int position, const VectorFloat & rhs);
		virtual void		   Set(int, int, const VectorFloat&)override;
		virtual void		   Set(int, int, float) {};

		virtual void		   Randu() {};
		virtual void		   Randn() {};
		virtual mDouble getAsMatrix() { return mDouble(); };
		virtual void SetRandom();
	};
}