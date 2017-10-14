#pragma once
#include "memory.h"
namespace gpuNN
{
	/// <summary>
	/// An vector for two integers
	/// </summary>
	class VectorInteger {
	
	public:
		/// <summary>
		/// Construct the vector and sets the values
		/// </summary>
		/// <param name="position">position</param>
		/// <param name="value">value</param>
		VectorInteger(int position, int value);
		/// <summary>
		/// Default constructor
		/// </summary>
		VectorInteger();
		/// <summary>
		/// CDopy Constructor
		/// </summary>
		/// <param name="integer">The rhs value</param>
		VectorInteger(const VectorInteger& integer);
		/// <summary>
		/// Destructor of vector
		/// </summary>
		~VectorInteger();
		/// <summary>
		/// Release the data
		/// </summary>
		void Release();
		/// <summary>
		/// Returns the value from the position
		/// </summary>
		/// <param name="position">The given position</param>
		/// <returns></returns>
		int Get(int position)const {
			return this->m_data[position];
		}
		/// <summary>
		/// Equal operator
		/// </summary>
		/// <param name="rhs">The rhs vector</param>
		/// <returns>The new vector</returns>
		VectorInteger& operator=(const VectorInteger& rhs);
		/// <summary>
		/// Initialize all the data with zero
		/// </summary>
		void Zeros();
		/// <summary>
		/// Initialize all the data with ones
		/// </summary>
		void Ones();
		/// <summary>
		/// Sets the value at given position
		/// </summary>
		/// <param name="position">The given position</param>
		/// <param name="value">The value</param>
		void Set(int position, int value);
		/// <summary>
		/// Sets all the elements with given value
		/// </summary>
		/// <param name="value">The given value</param>
		void SetAll(int value);
		/// <summary>
		/// Copy the *this element to the rhs vector
		/// </summary>
		/// <param name="rhs">The rhs vector</param>
		void CopyTo(VectorInteger& rhs)const;
		/// <summary>
		/// Overloading + operator
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>The result of operation</returns>
		VectorInteger operator+(const VectorInteger& rhs) const;
		/// <summary>
		/// Overloading - operator
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>The result of the operation</returns>
		VectorInteger operator-(const VectorInteger& rhs) const;
		/// <summary>
		/// Overloading * operator
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>The result of the operation</returns>
		VectorInteger operator*(const VectorInteger& rhs) const;
		/// <summary>
		/// Add rhs to each element from vector
		/// </summary>
		/// <param name="a">The given element</param>
		/// <returns>The vector result</returns>
		VectorInteger operator+(int rhs) const;
		/// <summary>
		/// Substract rhs from each element from the vector
		/// </summary>
		/// <param name="rhs">The element to substract</param>
		/// <returns>The result of the operation</returns>
		VectorInteger operator-(int rhs) const;
		/// <summary>
		/// 
		/// </summary>
		/// <param name="rhs"></param>
		/// <returns></returns>
		VectorInteger operator*(int rhs) const;
		/// <summary>
		/// Overloading operator +=
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>A reference of the result of the operation</returns>
		VectorInteger& operator+=(const VectorInteger& rhs);
		/// <summary>
		/// Overloading operator -=
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>A reference of the result of the operation</returns>
		VectorInteger& operator-=(const VectorInteger& rhs);
		/// <summary>
		/// Overloading operator *=
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>A reference of the result of the operation</returns>
		VectorInteger& operator*=(const VectorInteger& rhs);
		/// <summary>
		/// Overloading operator +=
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>A reference of the result of the operation</returns>
		VectorInteger& operator+=(int rhs);
		/// <summary>
		/// Overloading operator -=
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>A reference of the result of the operation</returns>
		VectorInteger& operator-=(int rhs);
		/// <summary>
		/// Overloading operator *=
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		/// <returns>A reference of the result of the operation</returns>
		VectorInteger& operator*=(int rhs);
		/// <summary>
		/// Multiply the vector with rhs parameter
		/// </summary>
		/// <param name="rhs">Ths rhs parameter</param>
		/// <returns>The result of the operation</returns>
		VectorInteger Mul(const VectorInteger& rhs) const;
		/// <summary>
		/// Multiply the vector with rhs parameter
		/// </summary>
		/// <param name="rhs">Ths rhs parameter</param>
		/// <returns>The result of the operation</returns>
		VectorInteger Mul(const int rhs) const;
		/// <summary>
		/// Allocates the vector
		/// </summary>
		void malloc();
		/// <summary>
		/// Prints the vector
		/// </summary>
		/// <param name="rhs">The rhs parameter</param>
		void Print(const std::string& rhs) const;

	protected:
		int* m_data;
	};

	/// <summary>
	/// Vector with three floats
	/// </summary>
	class VectorFloat {

	public:
		/// <summary>
		/// Default constructor
		/// </summary>
		VectorFloat();
		/// <summary>
		/// Constructor with three parameters
		/// </summary>
		/// <param name="a">The a parameter</param>
		/// <param name="b">The b parameter</param>
		/// <param name="c">The c parameter</param>
		VectorFloat(float a, float b, float c);
		/// <summary>
		/// Copy constructor
		/// </summary>
		/// <param name="rhs">The rhs object</param>
		VectorFloat(const VectorFloat& rhs);
		/// <summary>
		/// Destructor
		/// </summary>
		~VectorFloat();
		/// <summary>
		/// Release the data
		/// </summary>
		void Release();
		/// <summary>
		/// Equal operator =
		/// </summary>
		/// <param name="rhs"></param>
		/// <returns>The return parameter</returns>
		VectorFloat& operator=(const VectorFloat& rhs);
		/// <summary>
		/// Initialize the vector with zeros
		/// </summary>
		void Zeros();
		/// <summary>
		/// Initialize the vector with ones
		/// </summary>
		void Ones();
		/// <summary>
		/// Sets the position with value
		/// </summary>
		/// <param name="position">The position</param>
		/// <param name="value">The value</param>
		void Set(int position, float value);
		/// <summary>
		/// Sets all the positions with given values
		/// </summary>
		/// <param name="value">The given value</param>
		void SetAll(float value);
		/// <summary>
		/// Returns the value from the position
		/// </summary>
		/// <param name="position">The given position</param>
		/// <returns>The value</returns>
		float Get(int position) const;
		/// <summary>
		/// Copy the object to rhs vector
		/// </summary>
		/// <param name="rhs">The target vector</param>
		void  CopyTo(VectorFloat& rhs) const;

		VectorFloat operator+(const VectorFloat&) const;
		VectorFloat operator-(const VectorFloat&) const;
		VectorFloat operator*(const VectorFloat&) const;

		VectorFloat operator+(float) const;
		VectorFloat operator-(float) const;
		VectorFloat operator*(float) const;
		VectorFloat operator/(float) const;
		VectorFloat operator%(float) const;

		VectorFloat& operator+=(const VectorFloat&);
		VectorFloat& operator-=(const VectorFloat&);
		VectorFloat& operator*=(const VectorFloat&);

		VectorFloat& operator+=(const float);

		VectorFloat& operator-=(const float);

		VectorFloat& operator*=(const float);

		VectorFloat& operator/=(const float);
		VectorFloat& operator%=(const float);

		VectorFloat divNoRem(float) const;
		VectorFloat mul(const VectorFloat&) const;
		VectorFloat mul(const float) const;

		void malloc();

		void print(const std::string&) const;

	protected:
		float *m_data;

	};


}