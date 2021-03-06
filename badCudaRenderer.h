#ifndef __BAD_CUDA_RENDERER_H__
#define __BAD_CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"


class BadCudaRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numCircles;
    float* position;
    float* color;
    float* radius;

    float* cudaDevicePosition;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

public:

    BadCudaRenderer();
    virtual ~BadCudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name, bool benchMode);

    void allocOutputImage(int width, int height);

    void clearImage();

    void render();

    void shadePixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData);
};


#endif
