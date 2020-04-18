#!/bin/python3

# Handles converting New Leaf patterns to QR codes.

import numpy as np
import cv2
import pyqrcode

# (Debug option) Location to write raw QR data to
DATA_DUMP_PATH = "pattern-data.acnl"

# Converts a string to utf-16 encoded bytes, in little-endian, with no BOM.
# Output will be two bytes longer than the specified length due to the null-character placed at the end.
def _utf16_bytes(text, length):
	# Extend to length
	text = text.ljust(length, "\x00")

	# Truncate to length
	text = text[0 : length]

	# Append a null character
	text += "\x00"

	return [c for c in text.encode("utf-16LE")]

# Writes a field in the pattern data.
def _write_field(pattern_data, offset, length, value):
	# NumPy doesn't like assigning bytes objects to uint8 arrays, so convert them to a list of ints here.
	if isinstance(value, bytes):
		value = [b for b in value]

	pattern_data[offset : offset + length] = value

# Writes an id and a name to the pattern data.
# These types of identifiers are used to represent players and towns.
def _write_identifier(pattern_data, offset, id, name):
	_write_field(pattern_data, offset,        2, id)
	_write_field(pattern_data, offset + 0x2, 18, _utf16_bytes(name, 8))

# Convert pattern data to writable image data which OpenCV can use.
def to_img(qr):
	BORDER_SIZE = 3

	pixels = (1 - np.array(qr.code, dtype=np.uint8)) * 255
	height, width = pixels.shape
	image = np.zeros((height + BORDER_SIZE * 2, width + BORDER_SIZE * 2), dtype=np.uint8)
	image[:, :] = 255
	image[BORDER_SIZE : height + BORDER_SIZE, BORDER_SIZE : width + BORDER_SIZE] = pixels

	return image

# Export an already-generated QR as an image.
def export_img(filename, qr):
	cv2.imwrite(filename, to_img(qr))

# Create a QR code.
#
# param pattern_name  - Name of the pattern as a str. Will be truncated if longer than 20 characters.
# param palette       - Colour palette as iterable of 15 New Leaf palette ids.
# param indexed_image - 32*32 array of indices for each pixel in the image. Note that these indices are values
#                       0-14, indexing the given palette, not indices to New Leaf's global palette.
# param author_name   - Name of the author as a str. Will be truncated if longer than 8 characters.
# param town_name     - Name of the author's town as a str. Will be truncated if longer than 8 characters.
# param author_id     - Author's hidden id, as 2 bytes.
# param town_id       - Author's Town's hidden id, as 2 bytes.
# param unused_blocks - Bytes to write in unused blocks in the data. Any more than 10 bytes will be ignored.
#
# return - Tuple of:
#          - The raw QR data as a bytes string.
#          - A QR code in the format returned by pyqrcode.create.
def generate(pattern_name, palette, indexed_image, *,
		author_name="Pttn-Gen", town_name="Python", author_id=[105, 105], town_id=[4, 32], unused_blocks=b"BySyfPNHPG"):
	data = np.zeros(620, dtype=np.uint8)

	# Pattern name
	_write_field(data, 0x00, 42, _utf16_bytes(pattern_name, 20))

	# Author Identifier
	_write_identifier(data, 0x2a, author_id, author_name)

	# Unused block 2 bytes at 0x3e
	_write_field(data, 0x3e,  2, unused_blocks[0:2])

	# Author Town Identifier
	_write_identifier(data, 0x40, town_id, town_name)

	# Unused block 4 bytes at 0x54
	_write_field(data, 0x54,  4, unused_blocks[2:6])

	# Palette
	_write_field(data, 0x58, 15, palette)

	# Unused block 2 bytes at 0x67
	_write_field(data, 0x67,  2, unused_blocks[6:8])

	# Pattern Type - 0x09 = Normal pattern
	_write_field(data, 0x69,  1, 0x09)

	# Unused block 2 bytes at 0x6a
	_write_field(data, 0x6a,  2, unused_blocks[8:10])

	# Image
	# The image is stored with 2 pixels per byte.
	HEIGHT = 32
	WIDTH = 32

	encoded_pixels = np.zeros((HEIGHT, WIDTH // 2), dtype=np.uint8)
	for y in range(HEIGHT):
		for x in range(0, WIDTH, 2):
			px  = indexed_image[y, x]
			px |= indexed_image[y, x + 1] << 4
			encoded_pixels[y, x // 2] = px
	encoded_pixels = encoded_pixels.reshape(HEIGHT * WIDTH // 2)
	_write_field(data, 0x6c, HEIGHT * WIDTH // 2, encoded_pixels)

	# Convert data to QR
	byte_data = data.tobytes()
	qr_code = pyqrcode.create(byte_data, version=19, error="M", mode="binary", encoding="iso-8859-1")

	return byte_data, qr_code
