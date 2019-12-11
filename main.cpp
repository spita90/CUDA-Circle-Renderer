#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <GL/glut.h>

#include "refRenderer.h"
#include "cudaRenderer.h"
#include "badCudaRenderer.h"


void startRendererWithDisplay(CircleRenderer* renderer);
void CheckBenchmark(CircleRenderer* ref_renderer, CircleRenderer* cuda_renderer,
                    int totalFrames);


void usage(const char* progname) {
    printf("Utilizzo: %s [opzioni] nome_scena\n", progname);
    printf("I nome_scena validi sono: rand10k, rand100k\n");
    printf("Opzioni programma:\n");
    printf("  -r  --renderer <cpu/cuda>  Seleziona renderer: CPU oppure CUDA\n");
    printf("  -s  --size  <INT>          Dimensione rendering <INT>x<INT> pixel\n"
           "                                                    [deve essere multiplo di 32]\n");
    printf("  -b  --benchmark            Avvia modalita' benchmark comparando CPU e GPU\n");
    printf("  -m  --mode <good/bad>      Modalità render. CUDA: good - algoritmo corretto, oppure\n"
           "                                                    bad - algoritmo con race conditions\n"
           "                                                    [non si applica in modalità benchmark\n"
           "                                                     o con renderer cpu]\n");
    printf("  -?  --help                 Stampa questo messaggio\n");
}


int main(int argc, char** argv)
{

    int imageSize = 1024;

    std::string sceneNameStr;
    SceneName sceneName;
    bool useRefRenderer = true;
    bool benchMode = false;
    bool goodMode = true;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"renderer", 1, nullptr,  'r'},
        {"size",     1, nullptr,  's'},
        {"benchmark",0, nullptr,  'b'},
        {"mode",     1, nullptr,  'm'},
        {"help",     0, nullptr,  '?'},
        {nullptr ,0, nullptr, 0}
    };

    while ((opt = getopt_long(argc, argv, "r:s:m:b?", long_options, nullptr)) != EOF) {

        switch (opt) {
        case 'r':
            if (std::string(optarg).compare("cuda") == 0) {
                useRefRenderer = false;
            }
            break;
        case 's':
            if(atoi(optarg)%32!=0){
                printf("La dimensione deve essere multiplo di 32!");
                return 1;
            }else{
                imageSize = atoi(optarg);
                break;
            }
        case 'b':
            benchMode = true;
            break;
        case 'm':
            if (std::string(optarg).compare("bad") == 0) {
                goodMode = false;
                printf("IN THIS MODE YOU WILL HAVE GRAPHICAL GLITCHES\n"
                       "CAUSED BY DEVICE RACE CONDITIONS\n\n");
            }
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////


    if (optind + 1 > argc) {
        fprintf(stderr, "Errore: nome_scena mancante\n");
        usage(argv[0]);
        return 1;
    }

    sceneNameStr = argv[optind];
    if (sceneNameStr.compare("rand10k") == 0) {
        sceneName = CIRCLE_TEST_10K;
    } else if (sceneNameStr.compare("rand100k") == 0) {
        sceneName = CIRCLE_TEST_100K;
    } else {
        fprintf(stderr, "Il nome_scena e' sconosciuto (%s)\n", sceneNameStr.c_str());
        usage(argv[0]);
        return 1;
    }

    printf("Rendering immagine %dx%d pixel\n", imageSize, imageSize);

    CircleRenderer* renderer;

    if (benchMode) {
        // Need both the renderers

        CircleRenderer* ref_renderer;
        CircleRenderer* cuda_renderer;

        ref_renderer = new RefRenderer();
        cuda_renderer = new CudaRenderer();

        ref_renderer->allocOutputImage(imageSize, imageSize);
        ref_renderer->loadScene(sceneName, true);
        ref_renderer->setup();

        cuda_renderer->allocOutputImage(imageSize, imageSize);
        cuda_renderer->loadScene(sceneName, true);
        cuda_renderer->setup();

        // Check the correctness
        CheckBenchmark(ref_renderer, cuda_renderer, 5);

    } else {

        if (useRefRenderer)
            renderer = new RefRenderer();
        else{
            if(goodMode)
                renderer = new CudaRenderer();
            else
                renderer = new BadCudaRenderer();
        }
        renderer->allocOutputImage(imageSize, imageSize);
        renderer->loadScene(sceneName, false);
        renderer->setup();
        glutInit(&argc, argv);
        startRendererWithDisplay(renderer);
    }

    return 0;
}
