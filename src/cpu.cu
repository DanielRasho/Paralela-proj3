#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <png.h>

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 200;
const float radInc = degreeInc * M_PI / 180;
const int THRESHOLD = 200;  // Minimum votes required to consider a line

// ==========================================
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

// ==========================================
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

// ==========================================
// Draw a line on the RGB
void drawLine(unsigned char *image, int w, int h, float theta, float r) {
    // Pink color: RGB(255, 105, 180)
    const unsigned char pink_r = 255;
    const unsigned char pink_g = 105;
    const unsigned char pink_b = 180;

    int xCent = w / 2;
    int yCent = h / 2;

    // Draw line by iterating through image coordinates
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int xCoord = x - xCent;
            int yCoord = yCent - y;
            
            float calculated_r = xCoord * cos(theta) + yCoord * sin(theta);
            
            // If this pixel is close to the line, draw it in pink
            if (fabs(calculated_r - r) < 1.0) {
                int idx = y * w + x;
                image[idx * 3 + 0] = pink_r;
                image[idx * 3 + 1] = pink_g;
                image[idx * 3 + 2] = pink_b;
            }
        }
    }
}

// ==========================================
// The CPU Hough Transform
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  *acc = (int *)malloc(rBins * degreeBins * sizeof(int));
  memset (*acc, 0, sizeof (int) * rBins * degreeBins);
  
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  printf("Processing Hough Transform...\n");
  printf("  rMax = %.2f\n", rMax);
  printf("  rScale = %.2f\n", rScale);
  printf("  degreeBins = %d\n", degreeBins);
  printf("  rBins = %d\n\n", rBins);

  int edgePixels = 0;
  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
      {
        int idx = j * w + i;
        if (pic[idx] > 0)
          {
            edgePixels++;
            int xCoord = i - xCent;
            int yCoord = yCent - j;
            float theta = 0;
            for (int tIdx = 0; tIdx < degreeBins; tIdx++)
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (int)((r + rMax) / rScale);
                
                // Bounds checking
                if (rIdx >= 0 && rIdx < rBins) {
                  (*acc)[rIdx * degreeBins + tIdx]++;
                }
                theta += radInc;
              }
          }
      }
  
  printf("Processed %d edge pixels\n", edgePixels);
}

// ==========================================
int main (int argc, char **argv)
{
  if (argc < 3) {
    printf("Usage: %s <input.pgm> <output.png>\n", argv[0]);
    printf("Example: %s input.pgm output.png\n", argv[0]);
    printf("\nParameters:\n");
    printf("  Threshold: %d votes\n", THRESHOLD);
    printf("  Angle increment: %d degrees\n", degreeInc);
    printf("  Rho bins: %d\n", rBins);
    return 1;
  }

  printf("==== CPU-Only Hough Transform ====\n");
  printf("Input image: %s\n", argv[1]);
  printf("Output image: %s\n", argv[2]);
  printf("Threshold: %d votes\n\n", THRESHOLD);

  // Read PGM image
  PGMImage inImg;
  if (!readPGM(argv[1], &inImg)) {
    return 1;
  }

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  printf("Image dimensions: %d x %d\n", w, h);

  // Count edge pixels for debugging
  int edgeCount = 0;
  for (int i = 0; i < w * h; i++) {
    if (inImg.pixels[i] > 0) edgeCount++;
  }
  printf("Edge pixels detected: %d (%.2f%%)\n\n", edgeCount, 100.0 * edgeCount / (w * h));

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // Find lines above threshold and create output image
  printf("\n==== Detected Lines (threshold = %d votes) ====\n", THRESHOLD);
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
      int votes = cpuht[rIdx * degreeBins + tIdx];
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

  // Cleanup
  free(outputImage);
  free(cpuht);
  freePGM(&inImg);

  printf("\nDone!\n");
  return 0;
}