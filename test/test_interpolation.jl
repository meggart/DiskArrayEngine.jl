@testset "Interpolation" begin
    a = [i+j+k for i in 1:4, j in 1:5, k in 1:6]
    #source coordinates
    x = 5.0:5.0:20.0
    y = 2.0:3.0:14.0
    #target coordinates
    x2 = 5.0:0.5:20.0
    y2 = 1.5:1.0:14.5
    r = interpolate_diskarray(a,(1=>(x,x2),2=>(y,y2)))
end

@testset "Aggregate" begin
    using Statistics
    a = [i+j+k for i in 1:4, j in 1:5, k in 1:6]
    agg_mean = aggregate_diskarray(a, mean, (1=>nothing,))
    @test size(agg_mean) == (1,5,6)
    @test agg_mean[:,:,:] == mean(a, dims=1)
    agg_max = aggregate_diskarray(a, maximum, (2=>nothing,), strategy=:reduce)
    # This gives all ones for some reason
    @test_broken agg_max[:,:,:] == maximum(a, dims=2)
    agg_sec = aggregate_diskarray(a, mean, (2=>2,))
    # This should work but currently throws a bounds error
    @test_throws BoundsError agg_sec[:,:,:]
    agg_minimum = aggregate_diskarray(a, minimum, (3=>3,), strategy=:reduce)
    @test agg_minimum[:,:,1] == minimum(a, dims=3)[:,:]
    @test agg_minimum[:,:,2] == minimum(a, dims=3)[:,:] .+ 3
    @test_broken eltype(agg_minimum) == Int
    agg_minimum_direct = aggregate_diskarray(a, minimum, (3=>3,), strategy=:reduce)
    @test agg_minimum_direct[:,:,1] == minimum(a, dims=3)[:,:]
    @test agg_minimum_direct[:,:,2] == minimum(a, dims=3)[:,:] .+ 3
    @test_broken eltype(agg_minimum_direct) == Int
    # Why is the element type of minimum of an Int array Union{Missing, Float64}?
end