// Original atlas wrapper. Build against msdfgen core + ext/import-font.cpp and FreeType.
#include "msdfgen.h"
#include "ext/import-font.h"
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
int main(int argc, char **argv) {
    if (argc != 3) return 1;
    using namespace msdfgen;
    auto ft = initializeFreetype(); auto font = loadFont(ft, argv[1]);
    if (!font) return 2;
    std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const int w=32, h=48, cols=12, width=w*cols, height=h*3;
    std::vector<unsigned char> pixels(width*height*4, 0);
    for (int i=0; i<36; i++) {
        Shape shape; loadGlyph(shape, font, (unsigned)chars[i], FONT_SCALING_EM_NORMALIZED);
        shape.normalize(); auto bounds=shape.getBounds();
        edgeColoringSimple(shape, 3.0, 7);
        Bitmap<float, 3> bitmap(w, h);
        const double scale=36;
        Vector2 translate(w/(2*scale)-(bounds.l+bounds.r)/2, h/(2*scale)-(bounds.b+bounds.t)/2);
        generateMSDF(bitmap, shape, Projection(Vector2(scale),translate), Range(4/scale));
        for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
            int dst=(((i/cols)*h+(h-1-y))*width+(i%cols)*w+x)*4;
            for(int c=0;c<3;c++) pixels[dst+c]=(unsigned char)std::clamp(int(bitmap(x,y)[c]*255+0.5f),0,255);
            pixels[dst+3]=255;
        }
    }
    std::ofstream out(argv[2],std::ios::binary); out.write((char*)pixels.data(),pixels.size());
    destroyFont(font); deinitializeFreetype(ft);
}
