#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <chrono>
#include <thread>

#include <stdint.h>
#include <string.h>

#include <immintrin.h>


#define N       6000			// size of N*N matrix
#define DEBUG	false			// print matrixes (only suggested for small N)

#define INTRINSICS true

#define FALSE   0
#define TRUE    1


typedef float mrow_t[N + 1];		// wird benutzt um automatisch Pointer anzulegen 
typedef mrow_t* p_mrow_t;			// type for the matrix - 2d array



// Funktion zur Generierung und Intitalisierung einer Matrix mit Zufallswerten
// matrix ist zwei-dimensionales Array auf die Matrix
void initialize_system_random(p_mrow_t matrix) {
	int i, j;

	srand(0);
	for (i = 0; i < N; i++) {
		for (j = 0; j < N + 1; j++) {
			matrix[i][j] = (rand() % 100) + 1;
			//matrix[i][N] += (j - 50) * matrix[i][j];
		}
	}
}

void copy_matrix(p_mrow_t inMatrix, p_mrow_t outMatrix) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N + 1; j++) {
			outMatrix[i][j] = inMatrix[i][j];
		}
	}
}

void print_matrix(p_mrow_t matrix, bool unity) {
	printf("----------------------------------------\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N + 1; j++) {
			if (unity) {
				if (j == N) printf("= %4.1lf", matrix[i][j]);
				else printf("| %4.1lf ", matrix[i][j]);
			}
			else {
				if (j == N) printf("= %10.2lf", matrix[i][j]);
				else printf("| %10.2lf ", matrix[i][j]);
			}
		}
		printf("\n");
	}
	printf("----------------------------------------\n");
}

// to debug AVX variables
void print256_num(__m256 var)
{
	float val[8];
	memcpy(val, &var, sizeof(val));
	printf("Numerical: %f %f %f %f %f %f %f %f \n",
		val[0], val[1], val[2], val[3], val[4], val[5],
		val[6], val[7]);
}

p_mrow_t gauss_jordan_elimination(p_mrow_t matrix) {

	// init resulting matrix, which will be unit matrix in the end
	p_mrow_t matrix_intr;
	matrix_intr = (p_mrow_t)malloc(N * sizeof(mrow_t));
	copy_matrix(matrix, matrix_intr);

	int k, i, j;
	int increment = 8;
	for (k = 0; k < N; k++) {
		for (i = 0; i < N; i++) {
			for (j = k; j <= N;) {
				if (i != k) {
					if (!INTRINSICS || (j + increment) > (N + 1)) {
						float tmp = matrix[i][k] / matrix[k][k];
						matrix_intr[i][j] = matrix[i][j] - tmp * matrix[k][j];
						j++;
					}
					else {
						// using formula: matrix_intr[i][j] = matrix[i][j] - tmp * matrix[k][j];
						float tmp = matrix[i][k] / matrix[k][k];
						__m256 kj = _mm256_setr_ps(matrix[k][j], matrix[k][j + 1], matrix[k][j + 2], matrix[k][j + 3], matrix[k][j + 4], matrix[k][j + 5], matrix[k][j + 6], matrix[k][j + 7]);
						__m256 tmp8 = _mm256_setr_ps(tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp);
						__m256 ij = _mm256_setr_ps(matrix[i][j], matrix[i][j + 1], matrix[i][j + 2], matrix[i][j + 3], matrix[i][j + 4], matrix[i][j + 5], matrix[i][j + 6], matrix[i][j + 7]);
						__m256 result = _mm256_mul_ps(tmp8, kj);

						result = _mm256_sub_ps(ij, result);
						_mm256_store_ps(&matrix_intr[i][j], result);
						j += increment;
					}
				}
				else { // i == k, skip calculation, but increment j
					if (!INTRINSICS || (j + increment) > N) j++;
					else j += increment;
				}

			}
		}
	}

	// diagonal matrix -> unit matrix
	// its not possible to use intrinsics to modify matrix efficiently to unit matrix (values are not stored in line)
	// will use intrinsics to calculate results & store in different array
	for (int i = 0; i < N; i++) {
			matrix_intr[i][N] = matrix_intr[i][N] / matrix_intr[i][i];
			matrix_intr[i][i] = 1.0;
	}
	return matrix_intr;
}

int main(int argc, char** argv)
{
	p_mrow_t matrix;
	p_mrow_t matrix_intr;
	float* results = new float[N];
	int solution;
	clock_t start, finish;
	double duration;
	long sec, usec;

	printf("Anwenden des Gauss-Jordan-Algorithmus auf eine Matrix - parallel\n");
	printf("Dimensionen der Matrix: %dx%d\n", N, N);
	if (INTRINSICS) printf("Using AVX intrinsics with 256 bit operations (8x32 float)\n");
	printf("-----------------------------------------------------------------\n\n");

	// Speicher fuer Matrix allokieren, matrix mit Zufallswerten initialisieren
	matrix = (p_mrow_t)malloc(N * sizeof(mrow_t));
	initialize_system_random(matrix);
	if (DEBUG) print_matrix(matrix, false);

	// measure time and start calc
	start = clock();
	matrix_intr = gauss_jordan_elimination(matrix);
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;


	//console output
	//if (DEBUG) print_matrix(matrix_intr, true);
	if (DEBUG) {
		printf("Resulting vector:\n");
		for (int i = 0; i < N; i++) {
			printf("%4.1lf, ", results[i]);
		}
		printf("\n");
	}
	printf("Elapsed time for calc: %2.6f seconds\n", duration);

	return (0);
}