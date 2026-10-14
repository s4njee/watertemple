# Rebuilding the text atlas

The shipped atlas is genuine RGB MSDF data generated from Anton by msdfgen's
edgeColoringSimple + generateMSDF, with overlap/error correction enabled by its defaults.
Anton is licensed under OFL; its font and license are in public/fonts. The generator's
MIT license is here. The atlas contains A–Z and 0–9 in 12 columns of 32 × 48 px cells.
Glyphs are centred in each cell at 36 px/em; the signed distance range is four pixels.
Each glyph is flipped into top-left texture coordinates. Read the texture as linear data,
never sRGB. This demonstration uses fixed cell advances rather than kerning or shaping.

From the repository root, with C++17, pkg-config and FreeType installed:

```sh
git clone https://github.com/Chlumsky/msdfgen.git /tmp/watertemple-msdfgen
git -C /tmp/watertemple-msdfgen checkout 1c106ed8117893bf943e577f62eb0665fb271e46
clang++ -std=c++17 -O2 -DMSDFGEN_PUBLIC= -DMSDFGEN_USE_CPP11 \
  -I/tmp/watertemple-msdfgen $(pkg-config --cflags freetype2) \
  scripts/ui/generate-msdf.cpp /tmp/watertemple-msdfgen/core/*.cpp \
  /tmp/watertemple-msdfgen/ext/import-font.cpp \
  $(pkg-config --libs freetype2) -o /tmp/watertemple-msdfgen/atlas-gen
/tmp/watertemple-msdfgen/atlas-gen public/fonts/Anton-Regular.ttf /tmp/watertemple-ui-atlas.rgba
python3 scripts/ui/pack-atlas.py /tmp/watertemple-ui-atlas.rgba
npx prettier --write src/gpu/ui/atlas-data.ts
```

No native dependency or font download is needed at runtime. The gzip bytes are lazy-loaded
with the renderer and decompressed using DecompressionStream once per process. Each effect
owns and destroys its GPU texture. See https://github.com/Chlumsky/msdfgen for the method.
