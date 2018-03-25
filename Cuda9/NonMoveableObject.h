#pragma once

class NonMoveableObject {

public:
	NonMoveableObject() = default;
	~NonMoveableObject() = default;
public:
	NonMoveableObject(const NonMoveableObject&&) = delete;
	NonMoveableObject& operator=(const NonMoveableObject&&) = delete;
};