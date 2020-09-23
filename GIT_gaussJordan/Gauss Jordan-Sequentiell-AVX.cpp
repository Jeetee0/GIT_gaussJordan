/****************************************************************
*                                                               *
*  2.1. Gauss Jordan Elimination                                *
*                                                               *
*                                                               *
****************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <immintrin.h>

#define N       3000		// Groesse der N*N Matrix
#define DEBUG	false		// um Matrizen anzeigen zu lassen
#define EPSILON 1e-20
#define FALSE   0
#define TRUE    1

typedef float mrow_t[N + 1];     // wird benutzt um automatisch Pointer anzulegen 
typedef mrow_t* p_mrow_t;       // 2-dimensionales Array

// Funktion zur Generierung und Intitalisierung einer Matrix mit Zufallswerten
// matrix ist zwei-dimensionales Array auf die Matrix + eine spalte mit dem ergebnisvektor
void initialize_system(p_mrow_t matrix) {
	int i, j;

	srand(0);
	for (i = 0; i < N; i++) {
		for (j = 0; j <= N; j++) {
			if (j==N)
				matrix[i][j] = rand() * (rand() % 100) + 1;
			else 
				matrix[i][j] = rand();
		}
	}
}

void print_matrix(p_mrow_t matrix) {
	printf("----------------------------------------\n");
	for (int i = 0; i < N; i++) {
		printf("| ");
		for (int j = 0; j <= N; j++) {
			printf("%10.2lf | ", matrix[i][j]);
		}
		printf("\n");
	}
	printf("----------------------------------------\n");
}

// Gauss-Jordan-Eliminations-Funktion zu Loesung eines linearen Gleichungssystems
// Parameter: m - Zeiger auf Matrix
// Loesungsvektor in m[i][N]
int gauss_jordan_elimination(p_mrow_t m)
{
	int startrow, lastrow, nrow;
	float* pivotrow;
	float* resultsvector;
	int* pivotp, * marked;
	int i, j, k, picked;
	float tmp;


	/* rows of matrix I have to process */
	startrow = 0;
	lastrow = N;
	nrow = N;
	int increment = 8;

	//dynamische Speicherallokierung
	//Pivotrow: Reihe um naechste Reduzierung der Matrix durchzufuehren
	pivotrow = (float*)malloc((N + 1) * sizeof(float));

	resultsvector = (float*)malloc(N * sizeof(float));

	//pivotp: Speichert welche Spalte mit der Pivotreihe reduziert wurde
	pivotp = (int*)malloc(N * sizeof(int));
	//marked: Speichert welche Reihe schon fuer die Reduzierung benutzt wurde (keine Reihe doppelt)
	marked = (int*)malloc(nrow * sizeof(int));

	for (i = 0; i < nrow; i++)
		marked[i] = 0;

	for (i = 0; i < N; i++) {     // Spaltenzaehler
		//Finde Maximum von der aktuellen Spalte
		tmp = 0.0;
		for (j = 0; j < nrow; j++) {	// Zeilenzaehler

			//Wurde Reihe noch nicht als Pivotreihe verwendet und ist der aktuelle Wert der groesste
			if (!marked[j] && (fabs(m[j][i]) > tmp)) {
				tmp = fabs(m[j][i]);   //Speichere Betrag als Maximalwert
				picked = j;             //Speicher aus welcher Reihe der Max-Wert ist
			}
		}

		marked[picked] = 1;     //Markieren, dass die Reihe mit dem Maximal Element als Pivotreihe benutzt wird
		pivotp[picked] = i;     //Mit dieser Reihe, wird Spalte i reduziert

		//Pivotreihe laden
		for (j = 0; j < N + 1; j++)
			pivotrow[j] = m[picked][j];

		//Groesse der Pivotelemente ueberpruefen
		if (fabs(pivotrow[i]) < EPSILON) {       //Wenn Elemente zu klein dann keine Berechnung moeglich
			printf("Exits on iteration %d\n", i);
			return (FALSE);
		}

		//Reduziere alle Reihen mit der Pivotreihe
		for (j = 0; j < nrow; j++) {
			if (!(marked[j] && pivotp[j] == i)) {
				tmp = m[j][i] / pivotrow[i];
				for (k = i; k < N + 1; k++) {
					if ((k + increment) > (N + 1)) {
						m[j][k] -= pivotrow[k] * tmp;
						k++;
					}
					else {
						__m256 jk = _mm256_setr_ps(m[j][k], m[j][k + 1], m[j][k + 2], m[j][k + 3], m[j][k + 4], m[j][k + 5], m[j][k + 6], m[j][k + 7]);
						__m256 tmp8 = _mm256_setr_ps(tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp);
						__m256 pivotrow8 = _mm256_setr_ps(pivotrow[k], pivotrow[k + 1], pivotrow[k + 2], pivotrow[k + 3], pivotrow[k + 4], pivotrow[k + 5], pivotrow[k + 6], pivotrow[k + 7]);
						__m256 result = _mm256_mul_ps(tmp8, pivotrow8);

						result = _mm256_sub_ps(jk, result);
						_mm256_store_ps(&m[j][k], result);
						k += increment;
					}
				}
			}
		}
		if (DEBUG) printf("Durchlauf: %d\n", i);
		if (DEBUG) print_matrix(m);
	}

	// pivotpunkte ausgeben
	//for (i = 0; i < N; i++) {
	//	printf("%d, ", pivotp[i]);
	//}
	//printf("\n");
	

	//Loesungsvektor sortieren - Einheitsmatrix erstellen
	for (i = 0; i < N; i++) {         
			m[i][N] /= m[i][pivotp[i]];
			m[i][pivotp[i]] = 1.0;
			resultsvector[i] = m[i][N];
	}

	// diagonal anordnen
	for (i = 0; i < N; i++) {
		if (i != pivotp[i]) {
			m[pivotp[i]][pivotp[i]] = m[i][pivotp[i]];
			m[i][N] = resultsvector[i];
			m[i][pivotp[i]] = 0.0;
		}
		//printf("%f, ", resultsvector[i]);
	}
	printf("\n");
	return (TRUE);      //Berechnung erfolgreich
}

int main(int argc, char** argv)
{
	p_mrow_t a;
	int solution;
	clock_t start, finish;
	double duration;
	long sec, usec;

	printf("Anwenden des Gauss-Jordan-Algorithmus auf eine Matrix - sequentiell\n");
	printf("Dimensionen der Matrix: %dx%d\n", N, N);
	printf("-----------------------------------------------------------------\n\n");

	// Speicher fuer Matrix allokieren, matrix mit Zufallswerten initialisieren
	a = (p_mrow_t)malloc(N * sizeof(mrow_t));
	initialize_system(a);
	if (DEBUG) print_matrix(a);

	start = clock();
	solution = gauss_jordan_elimination(a);
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;

	if (solution == TRUE) {
		printf("\n\n\nEndergebnis:\n");
		if (DEBUG) print_matrix(a);
		printf("Elapsed time for calc: %2.6f seconds\n", duration);  // Zeitdauer ausgeben
	}
	else
		printf("No solution\n");

	return (0);
}