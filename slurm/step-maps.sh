# Dictionary mapping step names to numbers
declare -A step_nums
step_nums[setup-simulations]=1
step_nums[simulate-data]=2
step_nums[compute-covariance]=3
step_nums[allocate-points]=4
step_nums[initialize-loadings]=5
step_nums[tune-alpha]=6
step_nums[estimate-loadings]=7
step_nums[rotate]=8
step_nums[tune-sigmas]=9
step_nums[smooth-loadings]=10
step_nums[tune-kappa]=11
step_nums[shrink-loadings]=12
step_nums[compute-inv-err-cov]=13
step_nums[tune-gammas]=14
step_nums[estimate-factor-scores]=15
step_nums[melodic-data-prep]=16
step_nums[melodic-tune-sigma]=17
step_nums[melodic-estimation]=18

# Dictionary mapping step to number of CPUs
declare -A step_cpus
step_cpus[setup-simulations]=1
step_cpus[simulate-data]=1
step_cpus[compute-covariance]=2 # 20 # le-err: 2, le-bench: 20
step_cpus[allocate-points]=2
step_cpus[initialize-loadings]=1
step_cpus[tune-alpha]=2
step_cpus[estimate-loadings]=2 # le-err: 2, le-bench: 8
step_cpus[rotate]=1
step_cpus[tune-sigmas]=1
step_cpus[smooth-loadings]=1
step_cpus[tune-kappas]=1
step_cpus[shrink-loadings]=1
step_cpus[compute-inv-err-cov]=1
step_cpus[tune-gammas]=1
step_cpus[estimate-factor-scores]=1
step_cpus[melodic-data-prep]=1
step_cpus[melodic-tune-sigma]=1
step_cpus[melodic-estimation]=1

# Dictionary mapping step to memory (in GB) per CPU
declare -A step_mem_per_cpu
step_mem_per_cpu[setup-simulations]=1
step_mem_per_cpu[simulate-data]=20
step_mem_per_cpu[compute-covariance]=20
step_mem_per_cpu[allocate-points]=5
step_mem_per_cpu[initialize-loadings]=5
step_mem_per_cpu[tune-alpha]=5
step_mem_per_cpu[estimate-loadings]=5
step_mem_per_cpu[rotate]=5
step_mem_per_cpu[tune-sigmas]=5
step_mem_per_cpu[smooth-loadings]=5
step_mem_per_cpu[tune-kappas]=5
step_mem_per_cpu[shrink-loadings]=5
step_mem_per_cpu[compute-inv-err-cov]=20
step_mem_per_cpu[tune-gammas]=20 #5
step_mem_per_cpu[estimate-factor-scores]=20 #5
step_mem_per_cpu[melodic-data-prep]=5
step_mem_per_cpu[melodic-tune-sigma]=5
step_mem_per_cpu[melodic-estimation]=5

# Dictionary mapping step to walltime
declare -A step_time
step_time[setup-simulations]=01:00:00
step_time[simulate-data]=06:00:00 #06:00:00
step_time[compute-covariance]=16:00:00 #18:00:00
step_time[allocate-points]=06:00:00 #06:00:00
step_time[initialize-loadings]=12:00:00 #06:00:00
step_time[tune-alpha]=01:00:00 #48:00:00
step_time[estimate-loadings]=24:00:00 #12:00:00
step_time[rotate]=12:00:00
step_time[tune-sigmas]=32:00:00
step_time[smooth-loadings]=06:00:00
step_time[tune-kappas]=32:00:00
step_time[shrink-loadings]=06:00:00
step_time[compute-inv-err-cov]=12:00:00
step_time[tune-gammas]=24:00:00
step_time[estimate-factor-scores]=06:00:00
step_time[melodic-data-prep]=06:00:00 # TODO: Benchmark MELODIC steps
step_time[melodic-tune-sigma]=24:00:00
step_time[melodic-estimation]=06:00:00

