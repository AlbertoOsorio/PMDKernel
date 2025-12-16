using KernelAbstractions
using CUDA                    # optional, but needed for GPU test
using LinearAlgebra

# -------------------------------------------------------------
#  Kernel: fused dipole-field calculator (KernelAbstractions)
# -------------------------------------------------------------
@kernel function kernel_fused_B_KA!(
    R, P, M, B, n, m, μ0, BATCH_M
)
    i = @index(Global)
    valid = i <= n

    T = eltype(R)

    # Register accumulators
    Bx = zero(T); By = zero(T); Bz = zero(T)

    # Shared memory (6 × BATCH_M)
    shmem = @localmem(T, 6 * BATCH_M)

    # Local thread index
    lid = @index(Local)

    # Preload evaluation point
    Rx = valid ? R[i,1] : zero(T)
    Ry = valid ? R[i,2] : zero(T)
    Rz = valid ? R[i,3] : zero(T)

    jb = 1
    while jb <= m
        batch_size = min(BATCH_M, m - jb + 1)

        # --------------------------
        # Load M and P into shared mem
        # --------------------------
        if lid <= 6
            @inbounds for k in 1:batch_size
                if lid <= 3
                    shmem[(lid-1)*BATCH_M + k] =
                        M[lid, jb + k - 1]
                else
                    shmem[(lid-1)*BATCH_M + k] =
                        P[lid-3, jb + k - 1]
                end
            end
        end
        @synchronize()

        # --------------------------
        # Compute contributions
        # --------------------------
        if valid
            @inbounds for k in 1:batch_size
                # Load dipole moment
                μx = shmem[0*BATCH_M + k]
                μy = shmem[1*BATCH_M + k]
                μz = shmem[2*BATCH_M + k]

                # Position
                Px = shmem[3*BATCH_M + k]
                Py = shmem[4*BATCH_M + k]
                Pz = shmem[5*BATCH_M + k]

                # Relative vector
                dx = Rx - Px
                dy = Ry - Py
                dz = Rz - Pz

                # Dipole physics
                y  = dx*μx + dy*μy + dz*μz
                r2 = dx*dx + dy*dy + dz*dz
                r  = sqrt(r2)
                inv_r3 = one(T) / (r2 * r)
                inv_r5 = inv_r3 / r2
                scale = μ0 / (4*T(π))

                Bx += scale * (3*y*dx*inv_r5 - μx*inv_r3)
                By += scale * (3*y*dy*inv_r5 - μy*inv_r3)
                Bz += scale * (3*y*dz*inv_r5 - μz*inv_r3)
            end
        end

        @synchronize()
        jb += BATCH_M
    end

    # Output write (only if valid)
    if valid
        @inbounds B[i,1] = Bx
        @inbounds B[i,2] = By
        @inbounds B[i,3] = Bz
    end
end


# -------------------------------------------------------------
# CPU REFERENCE IMPLEMENTATION (slow but accurate)
# -------------------------------------------------------------
function reference_field(R, P, M, μ0)
    n = size(R,1)
    m = size(P,2)
    B = zeros(eltype(R), n, 3)

    for i in 1:n
        Rx,Ry,Rz = R[i,:]
        Bx=0;By=0;Bz=0

        for k in 1:m
            μx,μy,μz = M[:,k]
            Px,Py,Pz = P[:,k]

            dx = Rx - Px
            dy = Ry - Py
            dz = Rz - Pz

            y  = dx*μx + dy*μy + dz*μz
            r2 = dx*dx + dy*dy + dz*dz
            r  = sqrt(r2)
            inv_r3 = 1/(r2*r)
            inv_r5 = inv_r3/r2
            scale = μ0 / (4π)

            Bx += scale*(3*y*dx*inv_r5 - μx*inv_r3)
            By += scale*(3*y*dy*inv_r5 - μy*inv_r3)
            Bz += scale*(3*y*dz*inv_r5 - μz*inv_r3)
        end
        B[i,1]=Bx; B[i,2]=By; B[i,3]=Bz
    end

    return B
end


# -------------------------------------------------------------
# TEST PARAMETERS
# -------------------------------------------------------------
T = Float32
μ0 = T(1.0)
BATCH_M = 32

n = 1024          # evaluation points
m = 256           # dipoles

# Random test data
R = rand(T, n, 3)
P = rand(T, 3, m)
M = rand(T, 3, m)

# -------------------------------------------------------------
# Run REFERENCE
# -------------------------------------------------------------
println("Running reference...")
Bref = reference_field(R, P, M, μ0)

# -------------------------------------------------------------
# Choose backend from array type
# -------------------------------------------------------------
backend = KernelAbstractions.get_backend(R)

println("Using backend: ", backend)

# Move data if backend is GPU
if backend isa KernelAbstractions.CUDABackend
    println("Moving data to GPU...")
    R = CuArray(R)
    P = CuArray(P)
    M = CuArray(M)
    B = CuArray(zeros(T, n, 3))
else
    B = zeros(T, n, 3)
end

# -------------------------------------------------------------
# Launch KA kernel
# -------------------------------------------------------------
wg = BATCH_M
ndrange = n

println("Running KA kernel...")

kernel_fused_B_KA!(backend, ndrange; workgroupsize=wg)(
    R, P, M, B, n, m, μ0, BATCH_M
)

KernelAbstractions.synchronize(backend)

# Bring back result if needed
B_result = backend isa KernelAbstractions.CUDABackend ? Array(B) : B

# -------------------------------------------------------------
# Validate result
# -------------------------------------------------------------
err = maximum(abs.(B_result .- Bref))
println("Max error = ", err)
println(err < 1e-3 ? "PASS ✔" : "FAIL ✘")
