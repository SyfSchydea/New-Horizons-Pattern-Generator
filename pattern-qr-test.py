#!/bin/python3

# Handles converting New Leaf patterns to QR codes.
# Debug/test version. Should be adapted into an importable module later.

import numpy as np
import cv2
import pyqrcode

pattern_data = np.zeros(620, dtype=np.uint8)

def utf16_bytes(text, length):
	# Extend to length
	text = text.ljust(length, "\x00")

	# Truncate to length
	text = text[0 : length]

	# Append a null character
	text += "\x00"

	return [c for c in text.encode("utf-16LE")]

def write_field(offset, length, value):
	global pattern_data

	pattern_data[offset : offset + length] = value

# Export QR as image
def export_img(filename, qr):
	BORDER_SIZE = 3

	pixels = (1 - np.array(qr.code, dtype=np.uint8)) * 255
	height, width = pixels.shape
	image = np.zeros((height + BORDER_SIZE * 2, width + BORDER_SIZE * 2), dtype=np.uint8)
	image[:, :] = 255
	image[BORDER_SIZE : height + BORDER_SIZE, BORDER_SIZE : width + BORDER_SIZE] = pixels
	cv2.imwrite(filename, image)

# Pattern name
write_field(0x0, 42, utf16_bytes("Syf's pattern", 20))

# Author ID
write_field(0x2a, 2, [105, 105])

# Author Name
write_field(0x2c, 18, utf16_bytes("Syf", 8))

# Author Town ID
write_field(0x40, 2, [4, 32])

# Author Town Name
write_field(0x42, 18, utf16_bytes("Python", 8))

# Unused block 2 bytes at 0x56
hidden_text = b"SPNHPG"
write_field(0x56, 2, [c for c in hidden_text[0:2]])

# Palette
palette = [0x0F] * 15
palette[0] = 0x0f # white
palette[1] = 0xef # black
write_field(0x58, 15, palette)

# Unused block 2 bytes at 0x67
write_field(0x67, 2, [c for c in hidden_text[2:4]])

# Pattern Type
write_field(0x69, 1, 0x09)

# Unused block 2 bytes at 0x6a
write_field(0x6a, 2, [c for c in hidden_text[4:6]])

# Image
HEIGHT = 32
WIDTH = 32

pixels = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

pixels[:, :] = 0xf

pixels[1:3, 2  ] = 1
pixels[1:3, 4  ] = 1
pixels[4,   1  ] = 1
pixels[4,   5  ] = 1
pixels[5,   2:5] = 1

encoded_pixels = np.zeros((HEIGHT, WIDTH // 2), dtype=np.uint8)
for y in range(HEIGHT):
	for x in range(0, WIDTH, 2):
		px = pixels[y, x]
		px |= pixels[y, x + 1] << 4
		encoded_pixels[y, x // 2] = px
encoded_pixels = encoded_pixels.reshape(HEIGHT * WIDTH // 2)
write_field(0x6c, HEIGHT * WIDTH // 2, encoded_pixels)


byte_data = pattern_data.tobytes()

print(len(byte_data))

q = pyqrcode.create(byte_data, version=19, error="M", mode="binary", encoding="iso-8859-1")

with open("pattern.acnl", "wb") as f:
	f.write(byte_data)

export_img("pattern-qr.png", q)
