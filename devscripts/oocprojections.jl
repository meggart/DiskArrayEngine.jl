using Proj
using Plots
trans = Proj.Transformation("EPSG:4326", "+proj=utm +zone=32 +datum=WGS84", always_xy=true)

methods(trans)

xeurope = range(-10,35,step=0.05)
yeurope = range(35,65,step=0.05)

# xmat = first.(tuple.(xeurope,yeurope'))
# ymat = last.(tuple.(xeurope,yeurope'))

# scatter(vec(xmat),vec(ymat))

# t = trans.(tuple.(xeurope,yeurope'))
# scatter(vec(first.(t)),vec(last.(t)))


reduce_bb(x,y) = min(x[1],y[1]),max(x[2],y[1]),min(x[3],y[2]),max(x[4],y[2])

function edge_bb(trans,xin,yin)
    init = (Inf,-Inf,Inf,-Inf)
    for x in xin
        for y in (first(yin),last(yin))
            c = trans((x,y))
            init = reduce_bb(init,c)
        end
    end
    for y in yin
        for x in (first(xin),last(xin))
            c = trans((x,y))
            init = reduce_bb(init,c)
        end
    end
    init
end
outbox = edge_bb(trans,xeurope,yeurope)
target_x = range(outbox[1],outbox[2],length=1000)
target_y = range(outbox[3],outbox[4],length=1000)

itrans = inv(trans)
trans_step_size = (100,100)
using DiskArrays
xin = xeurope
yin = yeurope
procsteps = DiskArrays.GridChunks(length.((xin,yin)),trans_step_size)
outwindows = map(procsteps) do (xr,yr)
    bbout = edge_bb(trans,xeurope[xr],yeurope[yr])
end