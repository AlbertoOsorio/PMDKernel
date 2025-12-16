using GLMakie

function Slicer3D(fig, data;
                  colormap = :inferno,
                  colorrange = nothing,
                  zoom::Real = 1,
                  haircross = true,
                  pointvalue = true)

    sizesag, sizecor, sizeaxi = size(data)
    startpoint = Point3(round.(Int, size(data) ./ 2))

    crange = isnothing(colorrange) ? extrema(filter(!isnan, data)) : colorrange

    # --- Create a grid in the figure ---
    grid = fig[1, 1] = GridLayout()

    # --- Place axes (this creates rows/cols) ---
    aaxi = Axis(grid[1, 2], title = "Axial", aspect = DataAspect())
    asag = Axis(grid[2, 1], title = "Sagittal", aspect = DataAspect())
    acor = Axis(grid[2, 2], title = "Coronal", aspect = DataAspect())

    for ax in (aaxi, asag, acor)
        hidespines!(ax); hidedecorations!(ax)
    end

    # --- Place sliders and label (also create rows/cols) ---
    saxi = Slider(grid[1, 3], range = 1:sizeaxi, startvalue = startpoint[3]) # axial slider (right)
    scor = Slider(grid[2, 3], range = 1:sizecor, startvalue = startpoint[2]) # coronal slider (right)
    ssag = Slider(grid[3, 2], range = 1:sizesag, startvalue = startpoint[1]) # sagittal slider (bottom)
    lpvalue = Label(grid[3, 1], "", tellwidth = false, tellheight = false, halign = :left)

    # --- Now safe to set relative sizes (rows/cols exist now) ---
    rowsize!(grid, 1, Relative(0.45))  # top row (axial)
    rowsize!(grid, 2, Relative(0.45))  # middle row (sagittal + coronal)
    rowsize!(grid, 3, Relative(0.10))  # bottom row (sliders/label)

    colsize!(grid, 1, Relative(0.45))
    colsize!(grid, 2, Relative(0.45))
    colsize!(grid, 3, Relative(0.10))

    # --- Initial render ---
    heatmap!(aaxi, data[:, :, startpoint[3]], colormap = colormap, colorrange = crange)
    heatmap!(asag, data[startpoint[1], :, :]', colormap = colormap, colorrange = crange)
    heatmap!(acor, data[:, startpoint[2], :]', colormap = colormap, colorrange = crange)

    # --- Reactivity: update all three when any slider changes ---
    lift(ssag.value, scor.value, saxi.value) do x, y, z
        empty!(aaxi); empty!(asag); empty!(acor)

        heatmap!(aaxi, data[:, :, z], colormap = colormap, colorrange = crange)
        heatmap!(asag, data[x, :, :]', colormap = colormap, colorrange = crange)
        heatmap!(acor, data[:, y, :]', colormap = colormap, colorrange = crange)

        if haircross
            lines!(aaxi, [1, sizesag], [y, y], color = :white)
            lines!(aaxi, [x, x], [1, sizecor], color = :white)

            lines!(acor, [1, sizesag], [z, z], color = :white)
            lines!(acor, [x, x], [1, sizeaxi], color = :white)

            lines!(asag, [1, sizecor], [z, z], color = :white)
            lines!(asag, [y, y], [1, sizeaxi], color = :white)
        end

        if pointvalue
            val = data[x, y, z]
            lpvalue.text = isfinite(val) ? @sprintf("(%d, %d, %d) → %.3e", x, y, z, val) : "NaN"
        end
    end

    return (aaxi, asag, acor, saxi, ssag, scor)
end
