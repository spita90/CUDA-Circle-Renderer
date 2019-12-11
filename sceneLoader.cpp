
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <functional>
#include <algorithm>

#include "sceneLoader.h"

// randomFloat --
// //
// // return a random floating point value between 0 and 1
static float
randomFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
}

// generateRandomCircles --
// //
// // generates numCircles circles ordered by decreasing depth
static void
generateRandomCircles(
    unsigned int numCircles,
    float* position,
    float* color,
    float* radius,
    bool benchMode) {

    if(benchMode)
        srand(0); //always same seed: we`ll obtain same positions, colors, etc.
    else
        srand(static_cast<unsigned int>(time(nullptr)));  //seed always different
    std::vector<float> depths(numCircles);
    for (unsigned int i=0; i<numCircles; i++) {
        depths[i] = randomFloat();
    }

    std::sort(depths.begin(), depths.end(),  std::greater<float>());

    for (unsigned int i=0; i<numCircles; i++) {

        float depth = depths[i];

        radius[i] = .02f + .06f * randomFloat();

        unsigned int index3 = 3 * i;

        // positions are:
        // x: 0-left 1-right
        // y: 0-bottom 1-top
        position[index3] = randomFloat();
        position[index3+1] = randomFloat();
        position[index3+2] = depth;

        color[index3] = randomFloat();
        color[index3+1] = randomFloat();
        color[index3+2] = randomFloat();
    }
}

void
loadCircleScene(
    SceneName sceneName,
    int& numCircles,
    float*& position,
    float*& color,
    float*& radius,
    bool benchMode)
{
    if (sceneName == CIRCLE_TEST_10K) {

        // test scene containing 10K randomily placed circles

        numCircles = 10 * 1000;

        position = new float[3 * numCircles];
        color = new float[3 * numCircles];
        radius = new float[numCircles];

        generateRandomCircles(numCircles, position, color, radius, benchMode);

    } else if (sceneName == CIRCLE_TEST_100K) {

        // test scene containing 100K randomily placed circles

        numCircles = 100 * 1000;

        position = new float[3 * numCircles];
        color = new float[3 * numCircles];
        radius = new float[numCircles];

        generateRandomCircles(numCircles, position, color, radius, benchMode);

    } else {
        fprintf(stderr, "Errore: impossibile caricare scena (scena sconosciuta)\n");
        return;
    }

    printf("Caricata scena con %d cerchi\n", numCircles);
}
