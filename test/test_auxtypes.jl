@testset "ProductArrays" begin
    using DiskArrayEngine: ProductArray
    x1 = [1,2,3]
    x2 = ['a', 'b']
    x3 = [1.5,2.5,3.5,4.5]
    pa = ProductArray((x1,x2,x3))
    @test size(pa) == (3,2,4)
    @test eltype(pa) == Tuple{Int, Char, Float64}
    @test pa[1,1,1] == (1,'a',1.5)
    @test pa isa AbstractArray
end


@testset "LoopSplitter" begin
    using DiskArrayEngine: get_loopsplitter, split_loopranges_threads, merge_loopranges_threads, LoopWindows
    inow = (91:180,631:720,1:480)
    lspl = get_loopsplitter(3,((lw=LoopWindows(ProductArray((1:720,1:480)),Val((2,3))),chunks=(nothing, nothing),ismem=true),))

    @test lspl == DiskArrayEngine.LoopIndSplitter{(2,3),(1,),((false,1),(true,1),(true,2))}()

    tri, ntri = split_loopranges_threads(lspl,inow)

    i_thread = first(CartesianIndices(tri))
    i_nonthread = first(CartesianIndices(ntri))

    @test i_thread == CartesianIndex(631,1)
    @test i_nonthread == CartesianIndex(91)

    @test merge_loopranges_threads(i_thread,i_nonthread,lspl) == CartesianIndex(first.(inow))
end
