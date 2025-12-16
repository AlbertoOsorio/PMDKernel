using CUDA
using LinearAlgebra

const T = Float32 # Tune based on GPU shared memory
const μ0 = 4π * 1f-7  # Magnetic constant

function kernel_fused_B!(R, P, M, B, n, m)  # Added P parameter
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > n && return

    # Register accumulators (remain unchanged)
    Bx, By, Bz = zero(T), zero(T), zero(T)
    
    # Shared memory for BOTH moments and positions (6 × BATCH_M)
    # Layout: first 3 rows = moments, next 3 rows = positions
    shmem = CuStaticSharedArray(T, (6, BATCH_M))
    Msh = view(shmem, 1:3, :)
    Psh = view(shmem, 4:6, :)

    # Load evaluation point once into registers
    Rx, Ry, Rz = R[i, 1], R[i, 2], R[i, 3]

    jb = 1
    while jb <= m
        batch_size = min(BATCH_M, m - jb + 1)
        
        # Collaborative load: threads 1-6 load moments & positions
        if threadIdx().x <= 6
            @inbounds for k in 1:batch_size
                shmem[threadIdx().x, k] = (threadIdx().x <= 3) ? 
                    M[threadIdx().x, jb + k - 1] : 
                    P[threadIdx().x - 3, jb + k - 1]
            end
        end
        sync_threads()

        # Process each dipole in the batch
        @inbounds for k in 1:batch_size
            # Dipole moment
            μx, μy, μz = Msh[1, k], Msh[2, k], Msh[3, k]
            
            # RELATIVE VECTOR: r = R_eval - P_dipole
            dx = Rx - Psh[1, k]
            dy = Ry - Psh[2, k]
            dz = Rz - Psh[3, k]
            
            # Field calculation using relative vector
            y = dx*μx + dy*μy + dz*μz
            r2 = dx^2 + dy^2 + dz^2
            r = sqrt(r2)
            inv_r3 = 1f0 / (r2 * r)
            inv_r5 = inv_r3 / r2
            scale = μ0 / (4π)
            
            Bx += scale * (3f0 * y * dx * inv_r5 - μx * inv_r3)
            By += scale * (3f0 * y * dy * inv_r5 - μy * inv_r3)
            Bz += scale * (3f0 * y * dz * inv_r5 - μz * inv_r3)
        end
        
        sync_threads()
        jb += BATCH_M
    end

    @inbounds B[i, 1], B[i, 2], B[i, 3] = Bx, By, Bz
    return
end