@testset "Adjust candidates" begin
    optires = 205.3
intsizes = (30,40,50)
smax = 1_000_000
@test find_adjust_candidates(optires,smax,intsizes,max_order=2,reltol=0.01)==205//1
@test find_adjust_candidates(optires,smax,intsizes,max_order=2,reltol=0.05)==200//1

cand = find_adjust_candidates(optires,smax,intsizes,max_order=2,reltol=0.05)
ii = findfirst(ii->rem(cand.num,ii)==0,intsizes)



optires = 205.3
intsizes = (20,50)
smax = 1_000_000
@test find_adjust_candidates(optires,smax,intsizes,max_order=2,reltol=0.01)==205//1
@test find_adjust_candidates(optires,smax,intsizes,max_order=2,reltol=0.05)==200//1

optires = 1300
intsizes = (1000,)
smax=1_000_000
@test find_adjust_candidates(optires,smax,intsizes,max_order=2) == 1300//1
@test find_adjust_candidates(optires,smax,intsizes,max_order=3) == 4000//3

optires = 10
intsizes = (37,43)
smax = 200 
@test find_adjust_candidates(optires,smax,intsizes,max_order=2) == 86//9

optires = 124.3
intsizes = (365,)
smax=14610
find_adjust_candidates(optires,smax,intsizes,max_order=3) == 365//3

optires = 156.0
smax=720
intsizes=(90,)
@test find_adjust_candidates(optires,smax,intsizes) == 135//1
end

@testset "Loopwindows" begin
    using Dates, Test
    ircs = DiskArrays.IrregularChunks(chunksizes = daysinyear.(2000:2020))
    @test generate_LoopRange(365*5//3,ircs) == [1:366,367:975,976:1584,1585:2192,2193:2801,2802:3410,3411:4018,4019:4627,4628:5236,5237:5844,5845:6453,6454:7062,7063:7671]
    @test length(generate_LoopRange(365//3,ircs)) == 63
end