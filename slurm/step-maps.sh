# Dictionary mapping step names to numbers
declare -A step_nums
step_nums[setup-simulations]=1
step_nums[simulate-data]=2
step_nums[compute-covariance]=3
step_nums[allocate-points]=4
step_nums[initialize-loadings]=5
step_nums[tune-alpha]=6
step_nums[estimate-loadings]=7
step_nums[compute-inv-err-cov]=8
step_nums[tune-gamma]=9
step_nums[estimate-factor-scores]=10

# Dictionary mapping step to number of CPUs
declare -A step_cpus
step_cpus[setup-simulations]=1
step_cpus[simulate-data]=1
step_cpus[compute-covariance]=2
step_cpus[allocate-points]=2
step_cpus[initialize-loadings]=1
step_cpus[tune-alpha]=2
step_cpus[estimate-loadings]=2
step_cpus[compute-inv-err-cov]=1
step_cpus[tune-gamma]=1
step_cpus[estimate-factor-scores]=1

# Dictionary mapping step to memory (in GB) per CPU
declare -A step_mem_per_cpu
step_mem_per_cpu[setup-simulations]=1
step_mem_per_cpu[simulate-data]=20
step_mem_per_cpu[compute-covariance]=5
step_mem_per_cpu[allocate-points]=5
step_mem_per_cpu[initialize-loadings]=5
step_mem_per_cpu[tune-alpha]=5
step_mem_per_cpu[estimate-loadings]=5
step_mem_per_cpu[compute-inv-err-cov]=5
step_mem_per_cpu[tune-gamma]=5
step_mem_per_cpu[estimate-factor-scores]=5

# Dictionary mapping step to walltime ()
declare -A step_time
step_time[setup-simulations]=00:05:00
step_time[simulate-data]=01:00:00
step_time[compute-covariance]=02:00:00
step_time[allocate-points]=00:10:00
step_time[initialize-loadings]=00:10:00
step_time[tune-alpha]=00:10:00
step_time[estimate-loadings]=00:10:00
step_time[compute-inv-err-cov]=00:10:00
step_time[tune-gamma]=00:10:00
step_time[estimate-factor-scores]=00:10:00

