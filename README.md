# New Horizons Pattern Generator

This script converts images into patterns for Animal Crossing: New Horizons.

This is written in [Python 3](https://www.python.org/download/releases/3.0/) and
requires [NumPy](https://numpy.org/) and [OpenCV](https://opencv.org/).

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

## Instructions Files:

This file will show you how to draw the generated pattern.

### Colour palette:

eg.:

	Colour palette:
	        Hue:  6   8   4  11   3   7   4  11   3   4   7  11   4   2  15
	  Vividness:  5   7   6   9   6   9   7  11   7   6   6  12   5  12  10
	 Brightness: 11  10  13   6  14  12  14   9  14  14   8   7  13   8   5

<!-- TODO: Image examples of text output to NH palette screenshots -->

This section shows the colour palette to use to create this pattern. Each column
of numbers represents one colour in the palette. Each row represents one channel
in the colour space used by New Horizons. The numbers represent how far along each
slider to go, with `0` being the furthest left value on each slider. For hue, the
furthest right value is 29, while for vividness and brightness, the furthest right
value is 14.

### Pixel Maps:

eg.:

	Colour 3 [11  9  6]:
	  c c · · · · · · · c · c · c · ·  · · · · c c c · · · · · · · · c
	  · · · · · · · · · · · · · · · ·  a b a · · · · · c · c · · c · ·
	  · · · · c · c c c c · c c a b ·  · # # a · · · · · · · c · · · ·
	  · · · c c · c b · · · · · · · ·  · · · · · · · · · c · · · c · ·
	  · · · · c · a · · · · · · · · ·  · · · · · · · · · · c c c · c ·
	  · · · · · · b · · · · · · · · ·  # # # · · · · · · c c · c · · ·
	  · · · c · · a · · # · · · # # ·  · · # · · · · · · · c · c · c ·
	  c · c · · · · · · # · · · · · ·  · # # a · c c · · · · · · · c ·
	  c · c · · · · a · · · · · · · #  # # # · c · c c c · · · · · · c
	  · c · · · · · a · · · # # · # ·  · · · a c · · c · · · · · · · ·
	  · · c · · · · a · · · · · · · #  · # · · c · c · c · · · · · · ·
	  c · · · · · · c · · · · · · · ·  · · · · c · · c · · · · · · c ·
	  · · · · · · · c · · · · · · · ·  · # · # · · · · · · · · · · · ·
	  · · · · c c · a · · · · · · · ·  · · # # · c · · · · c · c · · ·
	  · · · · c · · a · · · · · · · ·  · # · # · # · c c · c · c · · ·
	  · · · · · c · · · · · · · · · b  · b b b b b # # a · · c · · · ·
	[...]

<!-- TODO: Image examples of pixel maps -> NH screenshots -->

The next section in the instructions file is a map of pixels for each colour in
the palette. Pixels represented as a dot (`·`) have not yet been filled by any
colour. Pixels represented by a hash (`#`) should be filled in with the current
colour. Pixels represented by a lower-case letter should have already been filled
by a previous colour. The first colour is represented by an `a`, the second colour
is represented by a `b`, and so on.

## Algorithm

To choose a palette, the script uses
[k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) on all the
colours present in the image. This chooses a good subset of colours to represent
the full set of colours.

Before performing k-means, the colours are converted to the
[Lab colour space](https://en.wikipedia.org/wiki/CIELAB_color_space) as this provides
a better representation of perceptual differences between colours.

After a palette has been generated, it is rounded to get an approximation of the palette,
which can be used in New Horizons. The palette is then used map each pixel in the image
to a colour in the palette.

## Dithering

<!-- TODO: Image examples of dithering vs no-dithering -->

[Dithering](https://en.wikipedia.org/wiki/Dither) may be used to simulate better colour variation, while creating a slightly
more noisy look to the pattern, which may be desirable or undesirable depending on
the image. The script can use [Floyd-Steinberg](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering)
dithering to achieve this effect in your patterns.

This option defaults to off, but may be toggled on using the `-d`/`--dithering` flag.

## Duplicate colours

Sometimes, when a palette is generated, two or more of the colours may be close
enough to each other that when converted to New Horizons' colour space, they are
rounded to exactly the same value. This is obviously undesirable, as this means
one colour is effectively going unused.

The current default policy to handle this situation is simply to output a warning
and continue regardless. Alternatively, you can pass the `-r`/`--retry-duplicate`
flag to automatically increment the RNG seed and retry palette generation until
a palette with no duplicates is created.

## Weight maps

A weight map may be used to give higher weights to some pixels when choosing palettes.
A weight map may be passed using the `-w`/`--weight-map`, and should be a greyscale
image with the same size as the main input image. Areas in the main image corresponding
to brighter values in the weight map will receive greater consideration than darker
areas when performing k-means to choose colours. This can be used to give more detail
to an area if you think it needs it.

## RNG Seed

The k-means algorithm does not generate the optimal set of clusters for its input data.
Doing so would be an [NP-hard](https://en.wikipedia.org/wiki/NP-hardness) problem.
Rather, it is a heuristic which aims to find a "good" solution in a reasonable time.
It also depends on a randomised starting state, and as such, may produce slightly
different results each time it is run. This starting state is determined by the
[RNG seed](https://en.wikipedia.org/wiki/Random_seed). The script generates and
prints a random seed before starting k-means. If you want to re-run the script exactly
as you previously ran it, you can pass a seed using `-s`/`--seed`.

<!-- TODO: Logging + -q/v -->
