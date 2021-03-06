#ifndef __SCENE_LOADER_H__
#define __SCENE_LOADER_H__

#include "circleRenderer.h"

void
loadCircleScene(
    SceneName sceneName,
    int& numCircles,
    float*& position,
    float*& color,
    float*& radius,
    bool benchMode);

#endif
