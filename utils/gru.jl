using GLMakie, Printf

function Slicer3D(fig,data;
                        colormap=:viridis,colorrange=nothing,
                        zoom::Int=1,
                        haircross=true,pointvalue=true)

    debug = true

    # Layout:
    # Data has dimensions B[x,y,z] x: R->L y: P->A z: F->H
    # --------------------------------
    # |    empty   | y  axial  |     |
    # |            |      x    |sl y |
    # |------------------------|-----|
    # | z sagittal | z coronal |     |
    # |      y     |     x     |sl z |
    # --------------------------------
    # |            |   sl x    |     |
    # --------------------------------

    # Obtain parameters from data
    sizesag,sizecor,sizeaxi = size(data)
    zsizesag,zsizecor,zsizeaxi = zoom.*size(data)
    startpoint = Point3(Int.(round.(size(data)./2)))
    if isnothing(colorrange) # Problem with NaN
        crange = extrema(filter(!isnan,data))
        # (minimum(data),maximum(data))
        # if isnan(crange[1])||isnan(crange[2])
        #     println("NaN in data, range set to (-1,1)")
        #     crange = (-1,1)
        # end
    else
        crange = colorrange
    end
    if debug
        println("size(data) = $(size(data))")
        println("crange = $crange")
    end

    # Create panels and sliders
        # fig = Figure(size=(650,500)) # Need to control the size
        aaxi = Axis(fig[1,2],aspect=DataAspect(),height=zsizecor,width=zsizesag)
        asag = Axis(fig[2,1],aspect=DataAspect(),height=zsizeaxi,width=zsizecor)
        acor = Axis(fig[2,2],aspect=DataAspect(),height=zsizeaxi,width=zsizesag)
        hidespines!(aaxi); hidedecorations!(aaxi)
        hidespines!(acor); hidedecorations!(acor)
        hidespines!(asag); hidedecorations!(asag)
        lpvalue = Label(fig[3,1],"")
        # Need to get rid of label (or position it in a better way)
        saxi = SliderGrid(fig[2,3],
                    (range=1:sizeaxi,startvalue=startpoint[3],horizontal=false,height=zsizeaxi))
        ssag = SliderGrid(fig[3,2],
                (range=1:sizesag,startvalue=startpoint[1],horizontal=true,width=zsizesag))
        scor = SliderGrid(fig[1,3],
            (range=1:sizecor,startvalue=startpoint[2],horizontal=false,height=zsizecor))

    # The interactive part
    @lift begin
        x = $(ssag.sliders[1].value)
        y = $(scor.sliders[1].value)
        z = $(saxi.sliders[1].value)
        heatmap!(asag,data[x,:,:],colormap=colormap,colorrange=crange)
        heatmap!(acor,data[:,y,:],colormap=colormap,colorrange=crange)
        heatmap!(aaxi,data[:,:,z],colormap=colormap,colorrange=crange)
        if haircross
            lines!(aaxi,[1;sizesag],[y;y],color=:white)
            lines!(aaxi,[x;x],[1;sizecor],color=:white)
            lines!(acor,[1;sizesag],[z;z],color=:white)
            lines!(acor,[x;x],[1;sizeaxi],color=:white)
            lines!(asag,[1;sizecor],[z;z],color=:white)
            lines!(asag,[y;y],[1;sizeaxi],color=:white)
        end
        if pointvalue
            lpvalue.text = @sprintf("(%d,%d,%d) -> %.4f",
                                    x,y,z,data[x,y,z])
        end
    end

    return saxi # returns the upper left grid to be used by the user

    # display(fig)

end