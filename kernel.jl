using CUDA
using LinearAlgebra

const scale = 1.0f-7 # μ0 / 4π

function _Bnu!(R, P, M, B, N, m)
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x    

    # Inicializamos acumulador de By
    Bx, By, Bz = 0.0f0, 0.0f0, 0.0f0
    
    # Cargamos punto de eval
    Rx, Ry, Rz = 0.0f0, 0.0f0, 0.0f0
    @inbounds if idx <= N
        Rx = R[idx, 1]
        Ry = R[idx, 2]
        Rz = R[idx, 3]
    end

    @inbounds if idx <= N
        # 1. Main Unrolled Loop
        # We step by 4. We stop at m - 3 to avoid going out of bounds.
        k = 1
        while k <= m - 3
            # --- Dipole 1 ---
            μx1, μy1, μz1 = M[1, k],   M[2, k],   M[3, k]
            px1, py1, pz1 = P[1, k],   P[2, k],   P[3, k]
            dx1, dy1, dz1 = Rx - px1, Ry - py1, Rz - pz1
            r2_1 = dx1*dx1 + dy1*dy1 + dz1*dz1
            
            # --- Dipole 2 ---
            μx2, μy2, μz2 = M[1, k+1], M[2, k+1], M[3, k+1]
            px2, py2, pz2 = P[1, k+1], P[2, k+1], P[3, k+1]
            dx2, dy2, dz2 = Rx - px2, Ry - py2, Rz - pz2
            r2_2 = dx2*dx2 + dy2*dy2 + dz2*dz2

            # --- Dipole 3 ---
            μx3, μy3, μz3 = M[1, k+2], M[2, k+2], M[3, k+2]
            px3, py3, pz3 = P[1, k+2], P[2, k+2], P[3, k+2]
            dx3, dy3, dz3 = Rx - px3, Ry - py3, Rz - pz3
            r2_3 = dx3*dx3 + dy3*dy3 + dz3*dz3

            # --- Dipole 4 ---
            μx4, μy4, μz4 = M[1, k+3], M[2, k+3], M[3, k+3]
            px4, py4, pz4 = P[1, k+3], P[2, k+3], P[3, k+3]
            dx4, dy4, dz4 = Rx - px4, Ry - py4, Rz - pz4
            r2_4 = dx4*dx4 + dy4*dy4 + dz4*dz4

            # --- Math Block 1 ---
            if r2_1 > 1.0f-18
                inv_r_1 = CUDA.rsqrt(r2_1)
                inv_r2_1 = inv_r_1 * inv_r_1
                inv_r3_1 = inv_r2_1 * inv_r_1
                inv_r5_1 = inv_r3_1 * inv_r2_1
                dot_mr_1 = dx1*μx1 + dy1*μy1 + dz1*μz1
                Bx += 3.0f0 * dot_mr_1 * dx1 * inv_r5_1 - μx1 * inv_r3_1
                By += 3.0f0 * dot_mr_1 * dy1 * inv_r5_1 - μy1 * inv_r3_1
                Bz += 3.0f0 * dot_mr_1 * dz1 * inv_r5_1 - μz1 * inv_r3_1
            end

            # --- Math Block 2 ---
            if r2_2 > 1.0f-18
                inv_r_2 = CUDA.rsqrt(r2_2)
                inv_r2_2 = inv_r_2 * inv_r_2
                inv_r3_2 = inv_r2_2 * inv_r_2
                inv_r5_2 = inv_r3_2 * inv_r2_2
                dot_mr_2 = dx2*μx2 + dy2*μy2 + dz2*μz2
                Bx += 3.0f0 * dot_mr_2 * dx2 * inv_r5_2 - μx2 * inv_r3_2
                By += 3.0f0 * dot_mr_2 * dy2 * inv_r5_2 - μy2 * inv_r3_2
                Bz += 3.0f0 * dot_mr_2 * dz2 * inv_r5_2 - μz2 * inv_r3_2
            end

            # --- Math Block 3 ---
            if r2_3 > 1.0f-18
                inv_r_3 = CUDA.rsqrt(r2_3)
                inv_r2_3 = inv_r_3 * inv_r_3
                inv_r3_3 = inv_r2_3 * inv_r_3
                inv_r5_3 = inv_r3_3 * inv_r2_3
                dot_mr_3 = dx3*μx3 + dy3*μy3 + dz3*μz3
                Bx += 3.0f0 * dot_mr_3 * dx3 * inv_r5_3 - μx3 * inv_r3_3
                By += 3.0f0 * dot_mr_3 * dy3 * inv_r5_3 - μy3 * inv_r3_3
                Bz += 3.0f0 * dot_mr_3 * dz3 * inv_r5_3 - μz3 * inv_r3_3
            end

            # --- Math Block 4 ---
            if r2_4 > 1.0f-18
                inv_r_4 = CUDA.rsqrt(r2_4)
                inv_r2_4 = inv_r_4 * inv_r_4
                inv_r3_4 = inv_r2_4 * inv_r_4
                inv_r5_4 = inv_r3_4 * inv_r2_4
                dot_mr_4 = dx4*μx4 + dy4*μy4 + dz4*μz4
                Bx += 3.0f0 * dot_mr_4 * dx4 * inv_r5_4 - μx4 * inv_r3_4
                By += 3.0f0 * dot_mr_4 * dy4 * inv_r5_4 - μy4 * inv_r3_4
                Bz += 3.0f0 * dot_mr_4 * dz4 * inv_r5_4 - μz4 * inv_r3_4
            end

            k += 4
        end

        # 2. Remainder Loop (Cleanup)
        # Handles the last 1, 2, or 3 items if m is not divisible by 4
        while k <= m
            μx, μy, μz = M[1, k], M[2, k], M[3, k]
            px, py, pz = P[1, k], P[2, k], P[3, k]
            dx, dy, dz = Rx - px, Ry - py, Rz - pz
            r2 = dx*dx + dy*dy + dz*dz

            if r2 > 1.0f-18
                inv_r = CUDA.rsqrt(r2)
                inv_r2 = inv_r * inv_r
                inv_r3 = inv_r2 * inv_r
                inv_r5 = inv_r3 * inv_r2
                dot_mr = dx*μx + dy*μy + dz*μz
                By += 3.0f0 * dot_mr * dy * inv_r5 - μy * inv_r3
                Bx += 3.0f0 * dot_mr * dx * inv_r5 - μx * inv_r3
                Bz += 3.0f0 * dot_mr * dz * inv_r5 - μz * inv_r3
            end
            k += 1
        end
    end
        
    # Write final result to Global Memory
    if idx <= N
        @inbounds B[idx, 1], B[idx, 2], B[idx, 3] = Bx * scale * 1000, By * scale * 1000, Bz * scale * 1000 # mT
    end
    return
end


function kernel_fused_B!(R, P, M, B, n, m)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    # Register accumulators for the result
    Bx, By, Bz = 0.0f0, 0.0f0, 0.0f0
    
    # Definimos shared memory
    shmem = CuStaticSharedArray(Float32, (6, BATCH_M))
    
    Rx, Ry, Rz = 0.0f0, 0.0f0, 0.0f0
    if i <= n
        Rx = R[i, 1]
        Ry = R[i, 2]
        Rz = R[i, 3]
    end

    jb = 1
    while jb <= m
        batch_size = min(BATCH_M, m - jb + 1)
        
        # Cargamos la batch de dipolos al espacio reservado de shared memory
        total_elements = 6 * batch_size
        tid = threadIdx().x
        while tid <= total_elements

            col = (tid - 1) ÷ 6 + 1
            row = (tid - 1) % 6 + 1
            
            global_col = jb + col - 1
            
            if row <= 3
                shmem[row, col] = M[row, global_col]
            else
                shmem[row, col] = P[row - 3, global_col]
            end
            tid += blockDim().x 
        end
        
        # Aseguramos que todos los datos finalizaron de ser cargados a shmem antes de comenzar los calculos
        sync_threads()

        
        if i <= n
            for k in 1:batch_size

                μx, μy, μz = shmem[1, k], shmem[2, k], shmem[3, k]
                px, py, pz = shmem[4, k], shmem[5, k], shmem[6, k]
                
                dx = Rx - px
                dy = Ry - py
                dz = Rz - pz
                
                r2 = dx*dx + dy*dy + dz*dz
                r = sqrt(r2)
                
                # Protect against r=0
                if r > 1.0f-9 
                    inv_r3 = 1.0f0 / (r2 * r)
                    inv_r5 = inv_r3 / r2
                    dot_mr = dx*μx + dy*μy + dz*μz
    
                    scale = 1.0f-7 # μ0 / 4π
    
                    Bx += scale * (3.0f0 * dot_mr * dx * inv_r5 - μx * inv_r3)
                    By += scale * (3.0f0 * dot_mr * dy * inv_r5 - μy * inv_r3)
                    Bz += scale * (3.0f0 * dot_mr * dz * inv_r5 - μz * inv_r3)
                end
            end
        end
        
        # Aseguramos que todos los threads terminen antes de la siguiente batch
        sync_threads()
        jb += BATCH_M
    end

    if i <= n
        B[i, 1], B[i, 2], B[i, 3] = Bx, By, Bz
    end
    return
end