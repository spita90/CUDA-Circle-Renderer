#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "sceneLoader.h"

// Costanti

#define ThrInBlock_X 32
#define ThrInBlock_Y 32
#define ThrInBlock (ThrInBlock_X*ThrInBlock_Y)
#define CirclesPerThread 16
#define MaxCircles 2048
#define SCAN_BLOCK_DIM (ThrInBlock)

#include "exclusiveScan.cu_inl"
#include "circleBoxTest.cu_inl"


////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU.
__constant__ GlobalConstants cuConstRendererParams;

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelRenderCircles -- (CUDA device code)
//
// ogni thread in un blocco analizza una parte dell'array di cerchi,
// fino ad avere, per ogni blocco, la lista degli indici dei cerchi
// che stanno su tale blocco.
// infine ogni thread renderizza un pixel dell'immagine finale.
//
__global__ void kernelRenderCircles() {

    __shared__ uint circleCount[ThrInBlock];
    __shared__ uint circleList[MaxCircles];
    __shared__ uint indexList[ThrInBlock];

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;

    //pre-calcolo gli inversi per futuri calcoli
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // a ogni blocco è associata una porzione quadrata
    // di dim. ThrInBlock_X x ThrInBlock_Y dell'immagine finale

    // indice lineare del thread nel blocco
    int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

    // coordinate blocco
    short blockLeftIdx = blockIdx.x * ThrInBlock_X;
    short blockRightIdx = blockLeftIdx + ThrInBlock_X - 1;
    short blockTopIdx = blockIdx.y * ThrInBlock_Y;
    short blockBottomIdx = blockTopIdx + ThrInBlock_Y - 1;

    // coordinate normalizzate blocco
    float blockLeftNormIdx = blockLeftIdx * invWidth;
    float blockRightNormIdx = blockRightIdx * invWidth;
    float blockTopNormIdx = blockTopIdx * invHeight;
    float blockBottomNormIdx = blockBottomIdx * invHeight;

    // ogni thread in ogni blocco analizza una segmentino dell'array di cerchi
    // quindi ogni blocco avrà analizzato l'intero array

    // calcolo dunque gli indici di competenza a ogni thread
    // il " + ThrInBlock - 1 " serve a compensare il troncamento a int
    int circlesPerThread = (cuConstRendererParams.numCircles + ThrInBlock - 1) / ThrInBlock;
    int circleStartIdx = threadIndex * circlesPerThread;
    int circleEndIdx = circleStartIdx + circlesPerThread;
    //l'ultimo thread prende tutti i restanti cerchi
    if(threadIndex == ThrInBlock)
        circleEndIdx = cuConstRendererParams.numCircles;

    int threadCircleCount = 0;
    uint threadCircleList[CirclesPerThread];
    for(int c = circleStartIdx; c<circleEndIdx; c++){
        // controllo sempre di non sbordare dall'array di cerchi
        if(c >= cuConstRendererParams.numCircles)
            break;
        float3 position = *(float3*)(&cuConstRendererParams.position[c * 3]);
        float radius = cuConstRendererParams.radius[c];
        // se il quadrato che comprende il cerchio ha almeno un punto
        // dentro il blocco corrente, mi salvo il suo indice
        if(circleInBoxConservative(position.x, position.y, radius, blockLeftNormIdx, blockRightNormIdx, blockBottomNormIdx, blockTopNormIdx) == 1){
            threadCircleList[threadCircleCount++] = c;
        }
    }

    // adesso threadCircleCount contiene, della parte dell'array di cerchi analizzati, quanti stanno nel blocco corrente,
    // e in threadCircleList, della parte dell'array di cerchi analizzati, gli indici di tali cerchi.
    circleCount[threadIndex] = threadCircleCount;
    __syncthreads();

    // circleCount è una successione che contiene per ogni thread il numero di cerchi (appartenenti
    // al blocco corrente) trovati nel rispettivo segmentino dell'array di tutti i cerchi.
    // Adesso metto in indexList la serie della successione circleCount
    // così da permettere poi ai thread di avere il corretto indice di partenza (privateIndex)
    // in circleList a partire da cui copiare gli indici dei cerchi relativi al loro segmentino
    //
    sharedMemExclusiveScan(threadIndex, circleCount, indexList, circleList, ThrInBlock);
    __syncthreads();

    uint privateIndex = indexList[threadIndex];

    for(int i=0; i<threadCircleCount; i++){
        circleList[privateIndex++] = threadCircleList[i];
    }
    __syncthreads();
    // la serie non contiene l'ultima somma (è esclusiva). La aggiungo ora per
    // ottenere il totale dei cerchi nel blocco
    uint totalCircles = indexList[ThrInBlock-1] + circleCount[ThrInBlock-1];

    // adesso che abbiamo la lista dei cerchi relativi al blocco corrente possiamo
    // iniziare il rendering vero e proprio
    uint pixelXCoord = blockLeftIdx + threadIdx.x;
    uint pixelYCoord = blockTopIdx + threadIdx.y;

    // calcolo la coordinata continua del centro del pixel
    // campionando poi il colore del pixel in base al fatto che il cerchio
    // includa effettivamente il centro del pixel o meno, ovvero
    // faccio ciò che in gergo CGI si definisce Point-Sampling
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelXCoord) + 0.5f),
        invHeight * (static_cast<float>(pixelYCoord) + 0.5f));
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelYCoord * imageWidth + pixelXCoord)]);
    float4 pixelData = *imgPtr;

    // adesso ogni thread renderizza un pixel, partendo dal cerchio più lontano.
    // Questa invariante è data dal fatto che i cerchi vengono già dati in ordine
    // di lontananza decrescente e mantenuta nella copia degli indici in circleList.
    for (uint i=0; i<totalCircles; i++){
        uint circleIndex = circleList[i];
        float3 position = *(float3*)(&cuConstRendererParams.position[circleIndex * 3]);

        float radius = cuConstRendererParams.radius[circleIndex];
        float diffX = position.x - pixelCenterNorm.x;
        float diffY = position.y - pixelCenterNorm.y;
        float pixelDist = diffX * diffX + diffY * diffY;
        float maxDist = radius * radius;
        // controllo se il cerchio è effettivamente sul pixel
        // o se lo era solo il quadrato che lo contiene
        if (pixelDist <= maxDist){
            float3 rgb;
            float alpha;
            int index3 = 3 * circleIndex;
            rgb = *(float3*)&(cuConstRendererParams.color[index3]);
            alpha = .5f;

            float oneMinusAlpha = 1.f - alpha;
            pixelData.x = alpha * rgb.x + oneMinusAlpha * pixelData.x;
            pixelData.y = alpha * rgb.y + oneMinusAlpha * pixelData.y;
            pixelData.z = alpha * rgb.z + oneMinusAlpha * pixelData.z;
            pixelData.w = alpha + pixelData.w;
        }
    }
    *imgPtr = pixelData;
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene, bool benchMode) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, color, radius, benchMode);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Inizializzo CUDA per il Renderer\n");
    printf("Trovati %d dispositivi CUDA\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);

        printf("Dispositivo %d: %s\n", i, deviceProps.name);
        printf("   Streaming Multiprocessors: %d\n", deviceProps.multiProcessorCount);
        printf("   Memoria Globale: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   Compute Capability:   %d.%d\n", deviceProps.major, deviceProps.minor);

        printf("   Dimensione Warp: %d\n", deviceProps.warpSize);

        printf("   Shared memory per blocco: %.0f KB\n", static_cast<float>(deviceProps.sharedMemPerBlock) / 1024);
        printf("   Shared memory per Str.Multipr.: %.0f KB\n", static_cast<float>(deviceProps.sharedMemPerMultiprocessor) / 1024);

        printf("   Registri per blocco: %d\n", deviceProps.regsPerBlock);
        printf("   Registri per Str. Mulitpr.: %d\n", deviceProps.regsPerMultiprocessor);

        printf("   Max threads per blocco: %d\n", deviceProps.maxThreadsPerBlock);
        printf("   Max threads per Str. Multipr.: %d\n", deviceProps.maxThreadsPerBlock);

    }
    printf("---------------------------------------------------------\n");


    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);

    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {

    dim3 blockDim(ThrInBlock_X, ThrInBlock_Y);
            dim3 gridDim((image->width + ThrInBlock_X - 1) / ThrInBlock_X,
                     (image->height + ThrInBlock_Y - 1) / ThrInBlock_Y);
            kernelRenderCircles<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}
