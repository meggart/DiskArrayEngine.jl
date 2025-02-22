using DiskArrays: eachchunk
export rechunk_diskarray

function copydata_rechunk!(xout, xin; threaded=true, dims=:)
    return xout .= xin
end

function rechunk_diskarray(aout, ain; max_cache=5e8, optimargs=(;))
    size(aout) == size(ain) ||
        throw(ArgumentError("Input and Output arrays must have the same size"))
    inar = InputArray(ain)

    outar = create_outwindows(size(aout); ismem=false, chunks=eachchunk(aout))

    f = create_userfunction(
        copydata_rechunk!, eltype(aout); is_blockfunction=true, is_mutating=true
    )

    op = GMDWop((inar,), (outar,), f)

    lr = optimize_loopranges(op, max_cache; optimargs...)

    r = DaggerRunner(op, lr, (aout,); workerthreads=false)

    return run(r)
end
