import matplotlib.pyplot as plt

# Dados das implementações
threads = [1, 2, 4, 8, 16]

# MPI
mpi_tempo = [19.553, 449.857, 288.378, 343.295, 545.989]
mpi_speedup = [1, 0.04, 0.07, 0.06, 0.04]
mpi_eficiencia = [100, 2, 1.75, 0.75, 0.25]

# CUDA
cuda_tempo = [30.293, 1.734, 681, 550, 567]
cuda_speedup = [1, 17.5, 44.5, 55.1, 53.4]
cuda_eficiencia = [100, 875, 1112.5, 688.8, 333.8]

# OpenMP
omp_tempo = [19.553, 10.127, 6.584, 4.973, 5.336]
omp_speedup = [1, 1.9, 3, 3.9, 3.7]
omp_eficiencia = [100, 95, 75, 48.8, 23.1]

# Criar gráficos
plt.figure(figsize=(15, 5))

# Gráfico de Tempo de Execução
plt.subplot(1, 3, 1)
plt.plot(threads, mpi_tempo, marker='o', label='MPI')
plt.plot(threads, cuda_tempo, marker='s', label='CUDA')
plt.plot(threads, omp_tempo, marker='^', label='OpenMP')
plt.xlabel("Número de Threads")
plt.ylabel("Tempo de Execução (s)")
plt.title("Tempo de Execução vs Threads")
plt.legend()
plt.grid()
plt.yscale("log")  

# Gráfico de Speedup
plt.subplot(1, 3, 2)
plt.plot(threads, mpi_speedup, marker='o', label='MPI')
plt.plot(threads, cuda_speedup, marker='s', label='CUDA')
plt.plot(threads, omp_speedup, marker='^', label='OpenMP')
plt.xlabel("Número de Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Threads")
plt.legend()
plt.grid()

# Gráfico de Eficiência
plt.subplot(1, 3, 3)
plt.plot(threads, mpi_eficiencia, marker='o', label='MPI')
plt.plot(threads, cuda_eficiencia, marker='s', label='CUDA')
plt.plot(threads, omp_eficiencia, marker='^', label='OpenMP')
plt.xlabel("Número de Threads")
plt.ylabel("Eficiência (%)")
plt.title("Eficiência vs Threads")
plt.legend()
plt.grid()

# Exibir os gráficos
plt.tight_layout()
plt.show()
