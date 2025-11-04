#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <png.h>

const int degreeInc = 2;
const int degreeBins = 90;  // 90
const int rBins = 136;
const float radInc = degreeInc * M_PI / 180;
const int THRESHOLD = 2100;

// ==========================================
// CONSTANT MEMORY 
__constant__ float d_Cos[90];  // degreeBins = 90
__constant__ float d_Sin[90];

//==============================
// Simple PGM reader
typedef struct {
    unsigned char *pixels;
    int x_dim;
    int y_dim;
    int max_gray;
} PGMImage;

int readPGM(const char *filename, PGMImage *img) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return 0;
    }

    char magic[3];
    fscanf(file, "%2s", magic);

    if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: Not a valid PGM file (P5 format)\n");
        fclose(file);
        return 0;
    }

    // Skip comments
    int c = getc(file);
    while (c == '#') {
        while (getc(file) != '\n');
        c = getc(file);
    }
    ungetc(c, file);

    // Read dimensions
    fscanf(file, "%d %d", &img->x_dim, &img->y_dim);
    fscanf(file, "%d", &img->max_gray);
    fgetc(file); // Read the newline

    // Allocate memory and read pixel data
    img->pixels = (unsigned char *)malloc(img->x_dim * img->y_dim);
    if (!img->pixels) {
        fprintf(stderr, "Error: Cannot allocate memory for image\n");
        fclose(file);
        return 0;
    }

    fread(img->pixels, 1, img->x_dim * img->y_dim, file);
    fclose(file);
    return 1;
}

void freePGM(PGMImage *img) {
    if (img->pixels) {
        free(img->pixels);
        img->pixels = NULL;
    }
}

//==============================
// PNG writer for output with colored lines
int writePNG(const char *filename, unsigned char *image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return 0;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return 0;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return 0;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return 0;
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    png_bytep row = (png_bytep)malloc(3 * width * sizeof(png_byte));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            row[x * 3 + 0] = image[idx * 3 + 0]; // R
            row[x * 3 + 1] = image[idx * 3 + 1]; // G
            row[x * 3 + 2] = image[idx * 3 + 2]; // B
        }
        png_write_row(png, row);
    }

    png_write_end(png, NULL);
    free(row);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return 1;
}

//==============================
// Draw a line on the RGB image (in pink color)
void drawLine(unsigned char *image, int w, int h, float theta, float r) {
    // Pink color: RGB(255, 105, 180)
    const unsigned char pink_r = 255;
    const unsigned char pink_g = 105;
    const unsigned char pink_b = 180;

    int xCent = w / 2;
    int yCent = h / 2;

    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int xCoord = x - xCent;
            int yCoord = yCent - y;

            float calculated_r = xCoord * cos(theta) + yCoord * sin(theta);

            if (fabs(calculated_r - r) < 1.0) {
                int idx = y * w + x;
                image[idx * 3 + 0] = pink_r;
                image[idx * 3 + 1] = pink_g;
                image[idx * 3 + 2] = pink_b;
            }
        }
    }
}

//==============================
// GPU kernel WITH SHARED MEMORY
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc,
                               float rMax, float rScale)
{
  // Define locID - local thread ID within the block
  int locID = threadIdx.x;
  
  // Calculate global thread ID
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;

  // LOCAL accumulator in shared memory
  __shared__ int localAcc[12240];  
  
  // Initialize local accumulator to 0
  // Each thread initializes multiple elements
  for (int i = locID; i < 12240; i += blockDim.x) {
    localAcc[i] = 0;
  }
  
  // First barrier: wait for all threads to finish initialization
  __syncthreads();

  // Boundary check
  if (gloID < w * h) {
    // Calculate center coordinates
    int xCent = w / 2;
    int yCent = h / 2;

    // Convert linear index to 2D coordinates and center them
    int xCoord = (gloID % w) - xCent;
    int yCoord = yCent - (gloID / w);

    // Only process edge pixels (non-zero pixels)
    if (pic[gloID] > 0)
    {
      // Vote for all possible lines through this pixel
      for (int tIdx = 0; tIdx < 90; tIdx++)
      {
        // Calculate rho using CONSTANT memory
        float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];

        // Convert rho to bin index
        int rIdx = (int)((r + rMax) / rScale);

        // Bounds check
        if (rIdx >= 0 && rIdx < 136)
        {
          // e) Update LOCAL accumulator (not global) with atomicAdd
          atomicAdd(&localAcc[rIdx * 90 + tIdx], 1);
        }
      }
    }
  }
  
  // Second barrier: wait for all threads to finish voting
  __syncthreads();
  
  // Transfer localAcc -> acc (global)
  // Each thread transfers multiple elements
  for (int i = locID; i < 12240; i += blockDim.x) {
    if (localAcc[i] > 0) {  // Only transfer if there are votes
      atomicAdd(&acc[i], localAcc[i]);
    }
  }
}

//==============================
int main (int argc, char **argv)
{
  // Check command line arguments
  if (argc < 3) {
    printf("Usage: %s <input.pgm> <output.png>\n", argv[0]);
    printf("Example: %s input.pgm output.png\n", argv[0]);
    printf("\nParameters:\n");
    printf("  Threshold: %d votes\n", THRESHOLD);
    printf("  Angle increment: %d degrees\n", degreeInc);
    printf("  Rho bins: %d\n", rBins);
    return 1;
  }

  printf("==== CUDA Hough Transform (SHARED + CONSTANT Memory) ====\n");
  printf("Input image: %s\n", argv[1]);
  printf("Output image: %s\n", argv[2]);
  printf("Threshold: %d votes\n\n", THRESHOLD);

  // Read PGM image
  PGMImage inImg;
  if (!readPGM(argv[1], &inImg)) {
    return 1;
  }

  int w = inImg.x_dim;
  int h = inImg.y_dim;

  printf("Image dimensions: %d x %d\n", w, h);

  // Count edge pixels for debugging
  int edgeCount = 0;
  for (int i = 0; i < w * h; i++) {
    if (inImg.pixels[i] > 0) edgeCount++;
  }
  printf("Edge pixels detected: %d (%.2f%%)\n\n", edgeCount, 100.0 * edgeCount / (w * h));

  // Pre-compute sin/cos values on host
  float *h_Cos = (float *)malloc(sizeof(float) * degreeBins);
  float *h_Sin = (float *)malloc(sizeof(float) * degreeBins);
  float rad = 0;
  for (int i = 0; i < degreeBins; i++)
  {
    h_Cos[i] = cos(rad);
    h_Sin[i] = sin(rad);
    rad += radInc;
  }

  // Calculate Hough space parameters
  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  printf("Hough Transform Parameters:\n");
  printf("  rMax = %.2f\n", rMax);
  printf("  rScale = %.2f\n", rScale);
  printf("  degreeBins = %d\n", degreeBins);
  printf("  rBins = %d\n", rBins);
  printf("  Accumulator size = %d bins\n\n", degreeBins * rBins);

  // Allocate device memory
  unsigned char *d_in;
  int *d_hough;

  cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
  cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);

  // Copy data to device
  cudaMemcpy(d_in, inImg.pixels, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
  
  // Use cudaMemcpyToSymbol for CONSTANT memory
  cudaMemcpyToSymbol(d_Cos, h_Cos, sizeof(float) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, h_Sin, sizeof(float) * degreeBins);
  
  // Initialize accumulator to zero
  cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

  // Setup kernel execution parameters
  int blockSize = 256;
  int gridSize = (w * h + blockSize - 1) / blockSize;

  // Calculate shared memory size
  int sharedMemSize = degreeBins * rBins * sizeof(int);

  printf("Memory Configuration:\n");
  printf("  Using CONSTANT memory for d_Cos and d_Sin (%d floats each)\n", degreeBins);
  printf("  Constant memory used: %lu bytes\n", 2 * sizeof(float) * degreeBins);
  printf("  Using SHARED memory for local accumulator\n");
  printf("  Shared memory per block: %d bytes (%.2f KB)\n\n", 
         sharedMemSize, sharedMemSize / 1024.0);

  printf("CUDA Kernel Configuration:\n");
  printf("  Block size: %d threads\n", blockSize);
  printf("  Grid size: %d blocks\n", gridSize);
  printf("  Total threads: %d\n\n", gridSize * blockSize);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record start event
  cudaEventRecord(start);

  // Launch kernel (d_Cos and d_Sin are in constant memory, no need to pass as parameters)
  printf("Launching kernel...\n");
  GPU_HoughTran<<<gridSize, blockSize>>>(d_in, w, h, d_hough, rMax, rScale);

  // Record stop event
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Kernel execution time: %.3f ms\n\n", milliseconds);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  cudaDeviceSynchronize();
  printf("Kernel completed!\n\n");

  // Copy results back to host
  int *h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));
  cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  printf("==== Detected Lines (threshold = %d votes) ====\n", THRESHOLD);
  int linesFound = 0;

  // Create RGB image for output (convert grayscale to RGB first)
  unsigned char *outputImage = (unsigned char *)malloc(w * h * 3);
  for (int i = 0; i < w * h; i++) {
    outputImage[i * 3 + 0] = inImg.pixels[i]; // R
    outputImage[i * 3 + 1] = inImg.pixels[i]; // G
    outputImage[i * 3 + 2] = inImg.pixels[i]; // B
  }

  // Find peaks in accumulator and draw lines
  for (int rIdx = 0; rIdx < rBins; rIdx++) {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
      int votes = h_hough[rIdx * degreeBins + tIdx];
      if (votes > THRESHOLD) {
        float theta = tIdx * radInc;
        float r = rIdx * rScale - rMax;

        printf("Line %d: theta=%.2f deg (%.4f rad), r=%.2f, votes=%d\n",
               linesFound, theta * 180.0 / M_PI, theta, r, votes);

        // Draw this line on the output image
        drawLine(outputImage, w, h, theta, r);
        linesFound++;
      }
    }
  }

  printf("\n==== Summary ====\n");
  printf("Total lines found: %d\n", linesFound);

  // Write output PNG
  if (writePNG(argv[2], outputImage, w, h)) {
    printf("Output image written to: %s\n", argv[2]);
  } else {
    fprintf(stderr, "Error writing output image\n");
  }

  // Cleanup host memory
  free(outputImage);
  free(h_hough);
  free(h_Cos);
  free(h_Sin);
  freePGM(&inImg);

  // Cleanup device memory
  cudaFree(d_in);
  cudaFree(d_hough);

  // Destroy CUDA events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("\nDone!\n");
  return 0;
}