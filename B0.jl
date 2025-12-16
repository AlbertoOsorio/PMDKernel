using NPZ
using Printf
using Distributions
using GPUArrays: @allowscalar
include("kernel.jl")
include("utils/gru.jl")
include("utils/bench.jl")

const BATCH_M =  1024

function B0(separation, viz::Bool, threads = 256)

    gx = -80:separation:80
    gy = gx
    gz = gx #[-110, -90, -70, -50, -30,-10, 10, 30, 50, 70, 90, 110, 130, 150]
    R_cpu = transpose(hcat([[x, y, z] for x in gx, y in gy, z in gz]...) )

    data = npzread("data/B0.npz")
    M1_cpu = data["array1"]
    M2_cpu = data["array2"]

    n = size(R_cpu, 1)
    m = size(M2_cpu, 2)

    B_cpu = zeros(Float32, n, 3)

    R = CuArray(R_cpu)
    M = CuArray(M2_cpu)
    P = CuArray(M1_cpu)
    B = CuArray(B_cpu)

    blocks = cld(n, threads)
    shmem = 6 * BATCH_M * sizeof(T) 

    if viz
        @cuda threads=threads blocks=blocks shmem=shmem kernel_fused_B!(R, P, M, B, n, m)
        B_res = Array(B')

        XX = [xi for xi in gx, yi in gy, zi in gz]
        mask = trues(size(XX))
        By = zeros(size(XX))
        @allowscalar begin
            By[mask] = B_res[2,:]
            fig = Figure(size=(600,600))
            saxi = Slicer3D(fig,By,zoom=1)  
            display(fig)
        end
    else
        benchmark_kernel(R, P, M, B, n, m, threads)
    end

#=
    println("Input R = ")
    display(R_cpu)

    println("\nInput M = ")
    display(M2_cpu)
    
    println("\nInput P = ")
    display(M1_cpu)

    println("\nOutput B = ")
    display(B_res)
=#
end