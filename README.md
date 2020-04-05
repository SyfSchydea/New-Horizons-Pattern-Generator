# New Horizons Pattern Generator

This script converts images into patterns for Animal Crossing: New Horizons.

This is written in Python 3 and requires NumPy and OpenCV.

<!-- TODO: Examples input/output images -->

## Basic Usage:

Call the script using:

	./convert-pattern.py path/to/your/image.png

Your image should be a 32×32 pixel image. It may be in any file-type which OpenCV is
able to open, which includes most common image formats such as `.jpg` and `.png`.

This will generate a preview image at `nh-pattern.png`, and a text file containing
instructions of how to produce the pattern at `nh-pattern-instructions.txt`.
The output location of the preview image may be overridden with the `-o`/`--out` option,
and the location of the instructions file may be overridden with the
`-i`/`--instructions-out` option. eg.:

	./convert-pattern.py input.png --out preview-output.png --instructions-out instructions-output.txt

<!-- TODO: How to read the instructions file -->

<!-- TODO: Overview of algorithm -->
<!-- TODO: Dithering -->
<!-- TODO: Duplicate colour handling -->
<!-- TODO: Weight maps -->
<!-- TODO: RNG Seed -->
<!-- TODO: Logging + -q/v -->
