#!/bin/python3

# Functions relating to colour spaces and conversions

import math

import numpy as np
import cv2

# Number of values per dimension in NH's colour space
NH_DEPTH_H = 30
NH_DEPTH_S = 15
NH_DEPTH_V = 15

# Range of HSV values which NH's colours space is able to represent
NH_RANGE_V_MIN = 0.05
NH_RANGE_V_MAX = 0.9

# Max values in OpenCV's HSV colour space
HSV_H_MAX = 360
HSV_S_MAX =   1
HSV_V_MAX =   1

# Distance between two Lab colours.
# Experimenting with using Delta E* 2000.
#
# param colour_a - First colour to compare.
# param colour_b - Second colour to compare.
#
# return         - Distance between the two colours.
def lab_distance(colour_a, colour_b):
	return math.sqrt(delta_e_2k_squared(colour_a, colour_b))

# Calculate Chroma (C*) from a* and b* values.
def _chroma(a, b):
	return math.sqrt(a * a + b * b)

# Calculate Hue (h*) from a* and b* values.
def _hue(a, b):
	if a == 0 and b == 0:
		return 0

	hue = math.atan2(b, a)

	# math.atan2 returns in the range -pi to pi radians.
	# But we need to convert to 0 to 360 degrees.
	if hue < 0:
		hue += math.tau
	
	return hue / math.tau * 360

# Sine in degrees
def _sin(x):
	return math.sin(x / 360 * math.tau)

# Cosine in degrees
def _cos(x):
	return math.cos(x / 360 * math.tau)

# The final step of calculating Delta E* is to take a square root. However this is a relatively expensive
# operation, and for many cases, unnecessary. So it is preferable to used this "squared" function where possible.
#
# This function was written using the Wikipedia page on the subject as reference. As such the variable
# name may make little sense out of context, but relate to the variables used on that page.
# https://en.wikipedia.org/wiki/Color_difference#CIEDE2000
def delta_e_2k_squared(colour_a, colour_b):
	# Unpack Lab values
	L_1, a_1, b_1 = colour_a
	L_2, a_2, b_2 = colour_b

	# L-prime - Lightness
	delta_L_prime = L_2 - L_1
	mean_L_prime = (L_1 + L_2) / 2

	# C* - Chroma
	C_1 = _chroma(a_1, b_1)
	C_2 = _chroma(a_2, b_2)
	mean_C = (C_1 + C_2) / 2

	# Evaluate coefficient used in a prime calculations to adjust for chromaticity
	mean_c_7 = mean_C ** 7
	c_coeff = math.sqrt(mean_c_7 / (mean_c_7 + 25 ** 7))
	c_coeff = (3 - c_coeff) / 2

	a_prime_1 = a_1 * c_coeff
	a_prime_2 = a_2 * c_coeff

	# C-prime - Adjusted Chroma
	C_prime_1 = _chroma(a_prime_1, b_1)
	C_prime_2 = _chroma(a_prime_2, b_2)

	mean_C_prime = (C_prime_1 + C_prime_2) / 2
	delta_C_prime = C_prime_2 - C_prime_1

	# h-prime - Adjusted Hue
	h_prime_1 = _hue(a_prime_1, b_1)
	h_prime_2 = _hue(a_prime_2, b_2)

	# Difference in Hue (delta-h-prime)
	# TODO: A lot of the conditionals in the calculations for delta and mean hue are the same. Merge them together
	hue_acute = abs(h_prime_1 - h_prime_2) <= 180
	if C_prime_1 == 0 or C_prime_2 == 0:
		delta_h_prime = 0
	else:
		delta_h_prime = h_prime_2 - h_prime_1
		if hue_acute:
			pass
		elif h_prime_2 <= h_prime_1:
			delta_h_prime += 360
		else:
			delta_h_prime -= 360
	
	delta_H_prime = 2 * math.sqrt(C_prime_1 * C_prime_2) * _sin(delta_h_prime / 2)

	# Mean Hue (H-bar)
	mean_H_prime = h_prime_1 + h_prime_2
	if hue_acute:
		pass
	elif mean_H_prime < 360:
		mean_H_prime += 360
	else:
		mean_H_prime -= 360

	# If either either chroma is 0, use the other colour's hue as the mean hue.
	if C_prime_1 != 0 and C_prime_2 != 0:
		mean_H_prime /= 2

	T = (1 - 0.17 * _cos(mean_H_prime     - 30)
	       + 0.24 * _cos(mean_H_prime * 2     )
	       + 0.32 * _cos(mean_H_prime * 3 +  6)
	       - 0.20 * _cos(mean_H_prime * 4 - 63))

	L_squared = (mean_L_prime - 50) ** 2
	S_L = 1 + 0.015 * L_squared / math.sqrt(20 + L_squared)
	S_C = 1 + 0.045 * mean_C_prime
	S_H = 1 + 0.015 * mean_C_prime * T

	mean_c_prime_7 = mean_C_prime ** 7
	R_T = -2 * math.sqrt(mean_c_prime_7 / (mean_c_prime_7 + 25 ** 7)) * _sin(60 * math.exp(-((mean_H_prime - 275) / 25) ** 2))

	lightness_dist = delta_L_prime / S_L
	chroma_dist    = delta_C_prime / S_C
	hue_dist       = delta_H_prime / S_H

	d_e_squared = (hue_dist ** 2
	             + chroma_dist ** 2
	             + hue_dist ** 2
	             + R_T * chroma_dist * hue_dist)

	return d_e_squared

# Convert a colour palette to a different colour space.
#
# param palette     - n*depth array of colours, where n is the
#                     number of colours in the palette.
# param conversions - Values which may be passed to cv2.cvtColor to
#                     define the input and output colour spaces.
#
# return            - Colour palette in the new colour space,
#                     in a similar format as the input.
def convert_palette(palette, *conversions):
	# cvtColour requires a 3d array, so reshape this into n*1*depth
	n, depth = palette.shape
	palette_img = np.reshape(palette, (n, 1, depth))
	for conversion in conversions:
		palette_img = cv2.cvtColor(palette_img, conversion)

	return np.reshape(palette_img, (n, depth))

# Convert a continuous saturation or brightness
# value to the closest NH palette value.
#
# param value           - Input saturation/brightness value.
# param max_input_value - Maximum value in the HSV colour space for the channel.
# param nh_range_min    - Lowest value which an NH colour can express in this channel.
# param nh_range_max    - Highest value which an NH colour can express in this channel.
def _quantise_channel(value, max_input_value, nh_range_min, nh_range_max, nh_depth):
	value /= max_input_value                 # Map to range 0-1
	value -= nh_range_min                    # Move min value to zero
	value /= (nh_range_max - nh_range_min)   # Move max value to one
	value = round(value * (nh_depth - 1))    # Round to nearest NH value
	value = min(max(value, 0), nh_depth - 1) # Clamp to bounds
	return value

# Convert NH saturation or brightness value to continuous HSV channels.
#
# param value        - Input saturation/brightness NH value
# param max_value    - Maximum value in the HSV colour space for the channel.
# param nh_range_min - Lowest value which an NH colour can express in this channel.
# param nh_range_max - Highest value which an NH colour can express in this channel.
def _nh_to_channel(value, max_value, nh_range_min, nh_range_max, nh_depth):
	value /= nh_depth - 1
	value *= nh_range_max - nh_range_min
	value += nh_range_min
	value *= max_value
	return value

# Fetch the maximum saturation of the NH colours, based on the brightness value.
# This is 100% for brightness values from 0 to 7,
# or between 100% and 97.5% for brightness values from 7 to 14
def nh_range_s_max(nh_brightness):
	if nh_brightness <= 7:
		return 1

	return (287 - nh_brightness) / 280

# Convert an HSV colour palette to New Horizon's subset of the HSV colour space.
#
# param palette - Array of colours in HSV.
# return        - Array of colours in New Horizon's colour space.
def palette_hsv_to_nh(palette):
	palette_size, depth = palette.shape
	nh_palette = np.zeros((palette_size, 3), dtype=np.int8)

	for i in range(palette_size):
		nh_colour = nh_palette[i]
		h, s, v = palette[i]

		nh_colour[0] = round(h / HSV_H_MAX *  NH_DEPTH_H) % NH_DEPTH_H
		nh_colour[2] = nh_v = _quantise_channel(v, HSV_V_MAX,
			NH_RANGE_V_MIN, NH_RANGE_V_MAX,       NH_DEPTH_V)
		nh_colour[1]        = _quantise_channel(s, HSV_S_MAX,
			0,              nh_range_s_max(nh_v), NH_DEPTH_S)

	return nh_palette

# Convert a palette of New Horizon colours to the HSV range used by OpenCV
def palette_nh_to_hsv(palette):
	palette_size, depth = palette.shape
	hsv_palette = np.zeros((palette_size, 3), dtype=np.float32)

	for i in range(palette_size):
		h, s, v    =     palette[i]
		hsv_colour = hsv_palette[i]

		hsv_colour[0] = h /  NH_DEPTH_H      * HSV_H_MAX
		hsv_colour[2] = _nh_to_channel(v, HSV_V_MAX,
			NH_RANGE_V_MIN, NH_RANGE_V_MAX,    NH_DEPTH_V)
		hsv_colour[1] = _nh_to_channel(s, HSV_S_MAX,
			0,              nh_range_s_max(v), NH_DEPTH_S)

	return hsv_palette

# Find duplicate colours in a palette.
#
# param palette - Colour palette as array
# return        - List of duplicate colours
def get_duplicate_colours(palette):
	# Allow either a 2d array of colours, or a 1d array of indices
	array_dims = len(palette.shape)
	if array_dims > 2:
		raise ValueError("get_duplicate_colours: parameter should be a 1 or 2 dimension array of colours")
	elif array_dims == 1:
		width, = palette.shape
		palette = palette.reshape((1, width))
	else:
		palette = palette.transpose()

	# Sort colours
	palette_record = np.core.records.fromarrays(palette, names="h, s, v")
	palette_record.sort()

	# Look for consecutive, identical colours
	palette_size, = palette_record.shape
	duplicate_colours = []
	for i in range(0, palette_size - 1):
		this_col = palette_record[i]
		# Log this colour is a duplicate if it matches the next one.
		# Skip adding it if it's already in the list of duplicates
		if (np.array_equal(this_col, palette_record[i + 1])
				and not (len(duplicate_colours) > 0
					and np.array_equal(this_col, duplicate_colours[-1]))):
			duplicate_colours.append(this_col)

	return duplicate_colours
