#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define N       2000			// size of N*N matrix
#define DEBUG	false			// print matrixes (only suggested for small N)
#define OPENMP	false			// toggle openMP parallelisation
#define THREADS 8				// number of threads for openMP


#define FALSE   0
#define TRUE    1


typedef double mrow_t[N + 1];		// wird benutzt um automatisch Pointer anzulegen 
typedef mrow_t* p_mrow_t;			// type for the matrix - 2d array



// Funktion zur Generierung und Intitalisierung einer Matrix mit Zufallswerten
// matrix ist zwei-dimensionales Array auf die Matrix
void initialize_system(p_mrow_t matrix) {
	int i, j;

	srand(0);
	for (i = 0; i < N; i++) {
		matrix[i][N] = 0.0;
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


p_mrow_t gauss_jordan_elimination(p_mrow_t matrix) {

	// init resulting matrix, which will be unit matrixin the end
	p_mrow_t matrix_new;
	matrix_new = (p_mrow_t)malloc(N * sizeof(mrow_t));
	initialize_system(matrix_new);

	int k, i, j;
	for (k = 0; k < N; k++) {
		#pragma omp parallel for private(i, j) num_threads(THREADS) if (OPENMP)
		for (i = 0; i < N; i++) { // do in parallel
			for (j = k; j <= N; j++) {
				if (i == k) continue;
				matrix_new[i][j] = matrix[i][j] - (matrix[i][k] / matrix[k][k]) * matrix[k][j];
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