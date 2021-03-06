#ifndef __REF_RENDERER_H__
#define __REF_RENDERER_H__

#include "circleRenderer.h"


class RefRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numCircles;
    float* position;
    float* color;
    float* radius;

public:

    RefRenderer();
    virtual ~RefRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name, bool benchMode);

    void allocOutputImage(int width, int height);

    void clearImage();

    void render();

    void shadePixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py,
        float* pixelData);
};


#endif
