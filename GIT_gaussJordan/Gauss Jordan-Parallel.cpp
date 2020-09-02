#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <chrono>
#include <thread>

#include <immintrin.h>


#define N       8			// size of N*N matrix
#define DEBUG	true			// print matrixes (only suggested for small N)

#define OPENMP	false			// toggle openMP parallelisation
#define THREADS 8				// number of threads for openMP

#define INTRINSICS true


#define FALSE   0
#define TRUE    1


typedef float mrow_t[N + 1];		// wird benutzt um automatisch Pointer anzulegen 
typedef mrow_t* p_mrow_t;			// type for the matrix - 2d array



// Funktion zur Generierung und Intitalisierung einer Matrix mit Zufallswerten
// matrix ist zwei-dimensionales Array auf die Matrix
void initialize_system(p_mrow_t matrix) {
	int i, j;

	srand(0);
	for (i = 0; i < N; i++) {
		for (j = 0; j < N+1; j++) {
			matrix[i][j] = rand();
			//matrix[i][N] += (j - 50) * matrix[i][j];
		}
	}
}

void print_matrix(p_mrow_t matrix, bool unity) {
	printf("----------------------------------------\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N+1; j++) {
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

void print_transposed_matrix(p_mrow_t matrix, bool unity) {
	printf("----------------------------------------\n");
	for (int i = 0; i < N+1; i++) {
		for (int j = 0; j < N; j++) {
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


p_mrow_t gauss_jordan_elimination(p_mrow_t matrix) {

	// using transposed matrix to make efficient use of intrinsics
	p_mrow_t transposed;
	transposed = (p_mrow_t)malloc((N+1) * sizeof(mrow_t));
	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N+1; ++j) {
			transposed[j][i] = matrix[i][j];
		}

	print_transposed_matrix(transposed, false);

	// init resulting matrix, which will be unit matrixin the end
	p_mrow_t matrix_new;
	matrix_new = (p_mrow_t)malloc(N * sizeof(mrow_t));

	int k, i, j;
	int increment = 8;
	for (k = 0; k < N; k++) {
		//#pragma omp parallel for private(i, j) num_threads(THREADS) if (OPENMP)
		for (i = 0; i < N; i++) { // do in parallel
			for (j = k; j <= N;) {
				if (i != k) {
					// using formula: matrix_new[i][j] = matrix[i][j] - tmp * matrix[k][j];
					float tmp = transposed[k][i] / transposed[k][k];
					__m256 kj = _mm256_setr_ps(matrix[j][k], matrix[j + 1][k], matrix[j + 2][k], matrix[j + 3][k], matrix[j + 4][k], matrix[j + 5][k], matrix[j + 6][k], matrix[j + 7][k]);
					__m256 tmp8 = _mm256_setr_ps(tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp);
					__m256 ij = _mm256_setr_ps(matrix[j][i], matrix[j + 1][i], matrix[j + 2][i], matrix[j + 3][i], matrix[j + 4][i], matrix[j + 5][i], matrix[j + 6][i], matrix[j + 7][i]);
					__m256 result = _mm256_mul_ps(tmp8, kj);

					result = _mm256_sub_ps(ij, result);

					_mm256_store_ps(&matrix_new[j][i], result);

					if (DEBUG) print_matrix(matrix_new, false);
					std::this_thread::sleep_for(std::chrono::milliseconds(1000));
				}
				j += increment;
			}
		}
	}

	//results/diagonal values
	for (int i = 0; i < N; i++) { // do in parallel
		matrix_new[i][N] = matrix_new[i][N] / matrix_new[i][i];
		matrix_new[i][i] = 1.0;
	}

	return matrix_new;
}

int main(int argc, char** argv)
{
	p_mrow_t matrix;
	p_mrow_t matrix_new;
	int solution;
	clock_t start, finish;
	double duration;
	long sec, usec;

	printf("Anwenden des Gauss-Jordan-Algorithmus auf eine Matrix - parallel\n");
	printf("Dimensionen der Matrix: %dx%d\n", N, N);
	if (OPENMP) printf("Using OpenMP with %d threads.\n", THREADS);
	printf("-----------------------------------------------------------------\n\n");

	// Speicher fuer Matrix allokieren, matrix mit Zufallswerten initialisieren
	matrix = (p_mrow_t)malloc(N * sizeof(mrow_t));
	initialize_system(matrix);
	if (DEBUG) print_matrix(matrix, false);

	// measure time and start calc
	start = clock();
	matrix_new = gauss_jordan_elimination(matrix);
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;


	//console output
	if (DEBUG) print_matrix(matrix_new, true);
	printf("Elapsed time for calc: %2.6f seconds\n", duration);

	return (0);
}