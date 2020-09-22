/****************************************************************
*                                                               *
*  2.1. Gauss Jordan Elimination                                *
*                                                               *
*                                                               *                                                              *
****************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N       3000     //groesse der N*N Matrix
#define EPSILON 1e-20
#define FALSE   0
#define TRUE    1

typedef double mrow_t[N+1];     //wird benutzt um automatisch Pointer anzulegen 
typedef mrow_t* p_mrow_t;       //2-dimensionales Arry

//Funktion zu Generierung und Intitalisierung einer Matrix mit Zufallswerten
//matrix ist 2 dimensionales Array auf die Matrix
void initialize_system (p_mrow_t matrix)
{
    int i, j;
 
    srand (0);
    for (i = 0; i < N; i++)
    {
        matrix[i][N] = 0.0;
        for (j = 0; j < N; j++)
        {
            matrix[i][j] = rand ();
            matrix[i][N] += (j - 50) * matrix[i][j];
        }
    }
}

//Ausgabe der Loesung fuer das Lineare Gleichungssystem
void print_results (p_mrow_t matrix)
{
    int i;          

    for (i = 0; i < N; i++)
    {
        printf ("x[%i] = %10.6lf\n", i, matrix[i][N]);
    }
}

//Gauss-Jordan-Eliminations-Funktion zu Loesung eines linearen Gleichungssystems
//Parameter: m - Zeiger auf Matrix
//Loesungsvektor in m[i][N]
int gauss_jordan_elimination (p_mrow_t m)
{
    int startrow, lastrow, nrow;        
    double *pivotrow;
    int *pivotp, *marked;
    int i, j, k, picked;
    double  tmp;
    
  
/* rows of matrix I have to process */
    startrow = 0;
    lastrow =  N;
    nrow=N;

//dynamische Speicherallocierung
//Pivotrow: Reihe um naechste Reduzierung der Matrix durchzufuehren
    pivotrow = (double *) malloc ((N+1) * sizeof (double));
//pivotp: Speichert welche Spalte mit der Pivotreihe reduziert wurde
    pivotp = (int *) malloc (N * sizeof (int));
//marked: Speicher welche Reihe fuer die Reduzierung benutzt wurde
    marked = (int *) malloc (nrow * sizeof (int));
 
    for (i = 0; i < nrow; i++)
        marked[i] = 0;

    for (i = 0; i < N; i++)     //Spaltenzaehler
    {             
            
        //Finde Maximum von der aktuellen Spalte
        tmp = 0.0;
        for (j = 0; j < nrow; j++)  //Zeilenzaehler
        {
            //Wurde Reihe noch nicht als Pivotreihe verwendet und ist der aktuelle Wert der groesste
            if (!marked[j] && (fabs (m[j][i]) > tmp))   
            {
                tmp = fabs (m[j][i]);   //Speichere Betrag als Maximalwert
                picked = j;             //Seicher aus welcher Reihe der Max-Wert ist
            }
        }

        marked[picked] = 1;     //Markieren, dass die Reihe mit dem Maximal Element als Pivotreihe benutzt wird
        pivotp[picked] = i;     //Mit dieser Reihe, wird Spalte i reduziert

        //Pivotreihe laden
        for (j = 0; j < N + 1; j++)
           pivotrow[j] = m[picked][j];      
        
        //Groesse der Pivotelemente ueberpruefen
        if (fabs (pivotrow[i]) < EPSILON)       //Wenn Elemente zu klein dann keine Berechnung moeglich
        {
            printf ("Exits on iteration %d\n", i);
            return (FALSE);
        }
//Reduziere alle Reihen mit der Pivotreihe
        for (j = 0; j < nrow; j++)
            if (!(marked[j] && pivotp[j] == i))
            {
                tmp = m[j][i] / pivotrow[i];
                for (k = i; k < N + 1; k++)
                    m[j][k] -= pivotrow[k] * tmp;
            }
    }
//Berechne die Loesung fuer die einzelnen Reihen
        for (i = 0; i < nrow; i++)
            m[i][N] = m[i][N] / m[i][pivotp[i]];

        for (i = 0; i < N; i++)         //Loesungsvektor sortieren
            m[pivotp[i]][0] = m[i][N];
        for (i = 0; i < N; i++)         //Loesung in m[i][N] speichern
            m[i][N] = m[i][0];
        
    return (TRUE);      //Berechnung erfolgreich
}

int main (int argc, char **argv)
{
    p_mrow_t a;
    int solution;
    clock_t start, finish;
    double duration;
    long sec, usec;

   
    //Spiecher fuer Matrix allocieren
    a = (p_mrow_t) malloc (N * sizeof(mrow_t));
    initialize_system (a);

    start=clock();      //Startzeit ermitteln

    solution = gauss_jordan_elimination (a);
    
    
    finish = clock();   //Endzeit ermitteln
    
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf( "%2.3f seconds\n", duration );  //Zeitdauer ausgeben

    //if (solution == TRUE)
    //    print_results (a);
    //else
    //    printf ("No solution\n");
    //
    return (0);
}