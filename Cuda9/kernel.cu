#include "includes.h"

class Person
{
public:
	int m_age;
	std::string m_name;
	std::string m_haircolor;
	Person(int m_age, std::string name, std::string hair) :
		m_age(m_age), m_name(name), m_haircolor(hair) {

	}
	~Person()
	{
		std::cout << "Destructorr called";
	}
	void print() {
		std::cout << m_name << " " << m_age << m_haircolor << std::endl;
	}


};

int main()
{
	
}
