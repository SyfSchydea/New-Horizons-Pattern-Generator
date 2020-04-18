#!/bin/python3

# Functions relating to colour spaces and conversions

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
# Currently just euclidean distance.
#
# param colour_a - First colour to compare.
# param colour_b - Second colour to compare.
#
# return         - Distance between the two colours.
def lab_distance(colour_a, colour_b):
	return np.linalg.norm(colour_a - colour_b)

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
