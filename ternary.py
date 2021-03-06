#!/bin/python3

# Implementation of ternary logic

# Ternary logic unit
# May be "true", "false", or "maybe"
class Trit(object):
	__slots__ = ["known", "value"]

	def __init__(self, known, value):
		super(Trit, self).__setattr__("known", known)
		super(Trit, self).__setattr__("value", value)
	
	# Disallow mutation
	def __setattr__(self, name, value):
		raise AttributeError("Trit is immutable")
	
	# Only allow value to be accessed if known is true
	def __getattribute__(self, name):
		known = super(Trit, self).__getattribute__("known")

		if name == "known":
			return known

		if name == "value":
			if known:
				return super(Trit, self).__getattribute__("value")
			else:
				raise ValueError("Value is not known")

		return super(Trit, self).__getattribute__(name)

	# Returns a bool to indicate if this Trit is known to be true.
	def definitely_true(self):
		return self.known and self.value

	# Returns a bool to indicate if this Trit is known to be false.
	def definitely_false(self):
		return self.known and not self.value

	# Returns a bool to indicate if this Trit might be true.
	def maybe_true(self):
		return not self.known or self.value

	# Returns a bool to indicate if this Trit might be false.
	def maybe_false(self):
		return not self.known or not self.value

	# Ambiguously take either this value or another.
	def maybe_set(self, other):
		other = Trit.of(other)

		if (self == other).definitely_true():
			return self

		return Trit.maybe

	# Check if two trits are equal
	def __eq__(self, other):
		# Coerce non-trits
		other = Trit.of(other)

		# If either trit is unknown, result is maybe
		if not self.known or not other.known:
			return Trit.maybe

		return Trit.of(self.value == other.value)

	def __ne__(self, other):
		return -(self == other)

	def definitely_equals(self, other):
		return (self == other).definitely_true()

	def definitely_not_equals(self, other):
		return (self == other).definitely_false()

	# Logical conjunction (AND)
	# Represented by the bitwise and operator (&)
	def __and__(self, other):
		if self.definitely_false():
			return Trit.false

		# Coerce non-trit values to Trits
		other = Trit.of(other)
		
		if other.definitely_false():
			return Trit.false

		if not self.known:
			return Trit.maybe

		return other

	def __rand__(self, other):
		return self.__and__(other)

	# Logical disjunction (OR)
	# Represented by the bitwise or operator (|)
	def __or__(self, other):
		if self.definitely_true():
			return Trit.true

		# Coerce non-trit values to Trits
		other = Trit.of(other)

		if other.definitely_true():
			return Trit.true

		if not self.known:
			return Trit.maybe

		return other

	def __ror__(self, other):
		return self.__or__(other)
	
	# Logical Inversion
	# Represented by the bitwise inversion operator (~)
	def __invert__(self):
		if not self.known:
			return Trit.maybe

		return Trit.of(not self.value)

	# Coerce this value to bool.
	# Will throw a value error if self is maybe
	def __bool__(self):
		return bool(self.value)

	def __str__(self):
		if not self.known:
			return "Maybe"

		return str(self.value)
	
	def __repr__(self):
		value_str = self.value if self.known else "None"
		return f"Trit({self.known}, {value_str})"

	# Always returns a Trit object
	@staticmethod
	def of(obj):
		if isinstance(obj, Trit):
			return obj

		return Trit(True, bool(obj))

Trit.true  = Trit(True,  True)
Trit.false = Trit(True,  False)
Trit.maybe = Trit(False, None)
