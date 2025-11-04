#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <png.h>

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;  // 90
const int rBins = 200;
const float radInc = degreeInc * M_PI / 180;
const int THRESHOLD = 2100;

// ==========================================
// CONSTANT MEMORY 
__constant__ float d_Cos[180 / 2];  // degreeBins = 90
__constant__ float d_Sin[180 / 2];

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
  __shared__ int localAcc[degreeBins * rBins];  // 90 * 200 = 18000 ints
  
  // Initialize local accumulator to 0
  // Each thread initializes multiple elements
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
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
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
      {
        // Calculate rho using CONSTANT memory
        float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];

        // Convert rho to bin index
        int rIdx = (int)((r + rMax) / rScale);

        // Bounds check
        if (rIdx >= 0 && rIdx < rBins)
        {
          // e) Update LOCAL accumulator (not global) with atomicAdd
          atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
        }
      }
    }
  }
  
  // Second barrier: wait for all threads to finish voting
  __syncthreads();
  
  // Transfer localAcc -> acc (global)
  // Each thread transfers multiple elements
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
    if (localAcc[i] > 0) {  // Only transfer if there are votes
      atomicAdd(&acc[i], localAcc[i]);
    }
  }
}