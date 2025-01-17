using Distributed

addprocs(8)

@everywhere function f(x)
    sleep(1)
    println(myid(), " ", x)
end

data = reshape(1:(6^3), 6, 6, 6)

function takeall!(res, c)
    while !isempty(c)
        push!(res, take!(c))
    end
end

function run_slice(data, workerlist)
    s = size(data)
    n = last(s)
    if length(data) > 10
        subworkerlists = [Channel{Int}(Inf) for _ in 1:n]
        newtasks = map(1:n) do i
            @async begin
                inds = map(_ -> :, size(data))
                inds = Base.setindex(inds, i, ndims(data))
                vdata = view(data, inds...)
                run_slice(vdata, subworkerlists[i])
            end
        end
        while true
            nexttask = (id=-1, nworkers=typemax(Int))
            anyrunning = false
            for i in eachindex(newtasks)
                t = newtasks[i]
                if istaskdone(t)
                    takeall!(workerlist, subworkerlists[i])
                else
                    anyrunning = true
                    if length(subworkerlists[i]) < nexttask.nworkers
                        nexttask = (id=i, nworkers=length(subworkerlists[i]))
                    end
                end
            end
            if !anyrunning
                break
            end
            worker = take!(workerlist)
            put!(subworkerlists[nexttask.id], worker)
        end
    else

    end




end


pool = CachingPool(Int[2, 3])

i = take!(pool)

put!(pool, i)