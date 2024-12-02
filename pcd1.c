#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 2000 // Tamanho da grade
#define T 500  // Número de iterações 
#define D 0.1  // Coeficiente de difusão
# define DELTA_T 0.01 //passo temporal
# define DELTA_X 1.0 //passo espacial

int NThreads = 2;       // Número de threads para OpenMP
double C[N][N] = {0};      // concentração inicial
double C_new[N][N] = {0};  // concentração para o próximo passo

void diff_eq(double C[N][N], double C_new[N][N]) {
    for (int t = 0; t < T; t++) {
        double difmedio = 0.0;
        #pragma omp parallel num_threads(NThreads)
        {
            #pragma omp for collapse(2)
            // equação de difusão
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    C_new[i][j] = C[i][j] + D * DELTA_T * (
                        (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                    );
                }
            }

            #pragma omp for collapse(2) reduction(+:difmedio)
            // atualizando a matriz para o próximo passo
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    difmedio += fabs(C_new[i][j] - C[i][j]);
                    C[i][j] = C_new[i][j];
                }
            }

            // Exibe a diferença média a cada 100 iterações igual dos prof
            //if ((t % 100) == 0)
              //  printf("Iteração %d - Diferença média = %g\n", t, difmedio / ((N-2) * (N-2)));
        }
    }
}

int main() {

    double start, end;
    //double **C = (double **)malloc(N * sizeof(double *));
    //double **C_new = (double **)malloc(N * sizeof(double *));
    // inicia com uma concentração alta no centro
    C[N/2][N/2] = 1.0;

    // executa a equação de difusão
    start = omp_get_wtime();
    diff_eq(C, C_new);
    end = omp_get_wtime();

    // resultados
    printf("Concentração final: %f ", C[N/2-2][N/2-2]);
    printf("\nTempo: %f ", end-start);
    printf("\nN Threads: %d\n", NThreads);
      

    return 0;
}
