#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 2000     // Tamanho da grade
#define T 500      // Número de iterações
#define D 0.1      // Coeficiente de difusão
#define DELTA_T 0.01   // Passo temporal
#define DELTA_X 1.0    // Passo espacial

void diff_eq(double **C, double **C_new, int local_n, int rank, int size) {
    double difmedio_local, difmedio_global;
    MPI_Status status;

    for (int t = 0; t < T; t++) {
        // Troca de bordas entre processos
        if (rank > 0) {
            MPI_Send(C[1], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
            MPI_Recv(C[0], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status);
        }
        if (rank < size-1) {
            MPI_Send(C[local_n], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
            MPI_Recv(C[local_n+1], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &status);
        }

        // Cálculo da difusão
        for (int i = 1; i <= local_n; i++) {
            for (int j = 1; j < N-1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) 
                    / (DELTA_X * DELTA_X)
                );
            }
        }

        // Cálculo da diferença média local
        difmedio_local = 0.0;
        for (int i = 1; i <= local_n; i++) {
            for (int j = 1; j < N-1; j++) {
                difmedio_local += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }

        // Redução para calcular a diferença média global
        MPI_Allreduce(&difmedio_local, &difmedio_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Impressão a cada 100 iterações
        //if (rank == 0 && t % 100 == 0) {
            //printf("Iteração %d - Diferença média = %g\n", 
              //     t, difmedio_global / ((N-2) * (N-2)));
        // }
    }
}

int main(int argc, char **argv) {
    int rank, size;
    double start, end;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Cálculo do tamanho local da grade
    int local_n = (N-2) / size;
    if (rank == size-1) {
        local_n += (N-2) % size;
    }
    
    // Alocação das matrizes locais (incluindo células fantasma)
    double **C = (double **)malloc((local_n + 2) * sizeof(double *));
    double **C_new = (double **)malloc((local_n + 2) * sizeof(double *));
    
    for (int i = 0; i < local_n + 2; i++) {
        C[i] = (double *)malloc(N * sizeof(double));
        C_new[i] = (double *)malloc(N * sizeof(double));
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            C_new[i][j] = 0.0;
        }
    }
    
    // Inicialização do ponto central no processo apropriado
    int mid_process = (N/2) / local_n;
    if (rank == mid_process) {
        int local_mid = (N/2) % local_n;
        C[local_mid][N/2] = 1.0;
    }
    
    // Sincronização antes de iniciar a medição do tempo
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    // Execução da difusão
    diff_eq(C, C_new, local_n, rank, size);
    
    // Sincronização antes de finalizar a medição do tempo
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    
    // Resultados
    if (rank == 0) {
        printf("\nResultados finais:\n");
        printf("Tempo de execução: %f segundos\n", end - start);
        printf("Número de processos: %d\n", size);
    }
    
    // Liberação da memória
    for (int i = 0; i < local_n + 2; i++) {
        free(C[i]);
        free(C_new[i]);
    }
    free(C);
    free(C_new);
    
    MPI_Finalize();
    return 0;
}