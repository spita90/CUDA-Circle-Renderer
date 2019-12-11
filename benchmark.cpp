#include <string>
#include <math.h>

#include "circleRenderer.h"
#include "cycleTimer.h"
#include "image.h"


static void compare_images(const Image* ref_image, const Image* cuda_image) {
    int i;
    int mismatch_count = 0;

    if (ref_image->width != cuda_image->width || ref_image->height != cuda_image->height) {
        printf ("Errore : dimensioni rendering differenti tra CPU e CUDA\n");
        printf ("Cuda : %dx%d\n", cuda_image->width, cuda_image->height);
        printf ("Cpu :  %dx%d\n", ref_image->width, ref_image->height);
        exit (1);
    }

    for (i = 0 ; i < 4 * ref_image->width * ref_image->height; i++) {
        // Compare with floating point error tolerance of 0.1f and ignore alpha
        if (fabs(ref_image->data[i] - cuda_image->data[i]) > 0.1f && i%4 != 3) {
            //could be only an aproximation error (aliasing), so check for four neighbours colour
            if ((fabs(ref_image->data[i-1*4] - cuda_image->data[i]) > 0.1f && i%4 != 3) &&
               (fabs(ref_image->data[i+1*4] - cuda_image->data[i]) > 0.1f && i%4 != 3) &&
               (fabs(ref_image->data[i-ref_image->width*4] - cuda_image->data[i]) > 0.1f && i%4 != 3) &&
               (fabs(ref_image->data[i+ref_image->width*4] - cuda_image->data[i]) > 0.1f && i%4 != 3)){
                //ok it is real the case
                mismatch_count++;

                /*
                // Get pixel number and print values
                int j = i/4;
                printf ("Rilevate differ. al pixel [%d][%d], valore %f, atteso %f ",
                        j/cuda_image->width, j%cuda_image->width,
                        cuda_image->data[i], ref_image->data[i]);

                printf ("per il canale ");
                switch (i%4) {
                    case 0 : printf ("rosso\n"); break;
                    case 1 : printf ("verde\n"); break;
                    case 2 : printf ("blu\n"); break;
                }
*/

            }

        }

        // Ignore 10 errors - those may come up because of rounding in distance calculation
        if (mismatch_count > 10) {
            printf ("ERRORE : Rilevate troppe differenze tra immagine giusta e rendering CUDA\n");
            printf ("ERRORE : Rendering CUDA non corretto!\n");
            exit (1);
        }

    }

    printf ("*********** Controllo correttezza rendering passato *******************\n");
}

void
CheckBenchmark(
    CircleRenderer* ref_renderer,
    CircleRenderer* cuda_renderer,
    int totalFrames)
{
    double totalRefClearTime = 0.f;
    double totalRefRenderTime = 0.f;
    double totalCudaClearTime = 0.f;
    double totalCudaRenderTime = 0.f;
    double totalTime = 0.f;

    printf("\nAvvio benchmark, %d frames...\n", totalFrames);

    double startTime = CycleTimer::currentSeconds();

    for (int frame=0; frame<totalFrames; frame++) {

        double startRefClearTime = CycleTimer::currentSeconds();
        ref_renderer->clearImage();
        double endRefClearTime = CycleTimer::currentSeconds();
        double startCudaClearTime = CycleTimer::currentSeconds();
        cuda_renderer->clearImage();
        double endCudaClearTime = CycleTimer::currentSeconds();

        double startRefRenderTime = CycleTimer::currentSeconds();
        ref_renderer->render();
        double endRefRenderTime = CycleTimer::currentSeconds();
        double startCudaRenderTime = CycleTimer::currentSeconds();
        cuda_renderer->render();
        double endCudaRenderTime = CycleTimer::currentSeconds();

        totalRefClearTime += endRefClearTime - startRefClearTime;
        totalRefRenderTime += endRefRenderTime - startRefRenderTime;
        totalCudaClearTime += endCudaClearTime - startCudaClearTime;
        totalCudaRenderTime += endCudaRenderTime - startCudaRenderTime;
    }

    compare_images(ref_renderer->getImage(), cuda_renderer->getImage());

    double endTime = CycleTimer::currentSeconds();
    totalTime = endTime - startTime; 

    printf("Tempo complessivo:  %.4f sec\n", totalTime);
    printf("\n");
    printf("Tempi medi per frame:\n");
    printf("Clear  - CPU: %.2f ms       GPU: %.2f ms\n", (1000.f * totalRefClearTime / totalFrames), (1000.f * totalCudaClearTime / totalFrames));
    printf("Render - CPU: %.2f ms    GPU: %.2f ms\n", (1000.f * totalRefRenderTime / totalFrames), (1000.f * totalCudaRenderTime / totalFrames));
    printf("Totale - CPU: %.2f ms    GPU: %.2f ms\n", ((1000.f * totalRefClearTime / totalFrames))+(1000.f * totalRefRenderTime / totalFrames),
           (1000.f * totalCudaClearTime / totalFrames)+(1000.f * totalCudaRenderTime / totalFrames));
    printf("\n");
    printf("Fattore di Speedup ottenuto: %.4f\n",(((1000.f * totalRefClearTime / totalFrames))+(1000.f * totalRefRenderTime / totalFrames))/
           ((1000.f * totalCudaClearTime / totalFrames)+(1000.f * totalCudaRenderTime / totalFrames)));
}
