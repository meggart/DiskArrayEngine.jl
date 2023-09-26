using Test

@testset "Window types" begin
using DiskArrayEngine: get_ordering, get_overlap, get_sparsity
w1 = DiskArrayEngine.to_window(collect(1:100))
@test get_ordering(w1) == DiskArrayEngine.Increasing()
@test get_overlap(w1) == DiskArrayEngine.NonOverlapping()
@test get_sparsity(w1) == DiskArrayEngine.Dense()
@test w1[2] == 2

w1 = DiskArrayEngine.to_window(1:100)
@test w1 isa UnitRange
@test get_ordering(w1) == DiskArrayEngine.Increasing()
@test get_overlap(w1) == DiskArrayEngine.NonOverlapping()
@test get_sparsity(w1) == DiskArrayEngine.Dense()
@test w1[2] == 2

w2 = DiskArrayEngine.to_window(collect(100:-1:1));
@test get_ordering(w2) == DiskArrayEngine.Decreasing()
@test get_overlap(w2) == DiskArrayEngine.NonOverlapping()
@test get_sparsity(w2) == DiskArrayEngine.Dense()
@test w2[2] == 99

w2 = DiskArrayEngine.to_window(100:-1:1);
@test w2 isa StepRange
@test get_ordering(w2) == DiskArrayEngine.Decreasing()
@test get_overlap(w2) == DiskArrayEngine.NonOverlapping()
@test get_sparsity(w2) == DiskArrayEngine.Dense()
@test w2[2] == 99

w3 = DiskArrayEngine.MovingWindow(10,1,2,100)
@test get_ordering(w3) == DiskArrayEngine.Increasing()
@test get_overlap(w3) == DiskArrayEngine.Overlapping()
@test get_sparsity(w3) == DiskArrayEngine.Dense()
@test w3[2] == 11:12
@test DiskArrayEngine.to_window(w3) === w3

w4 = DiskArrayEngine.to_window(collect(w3))
@test get_ordering(w4) == DiskArrayEngine.Increasing()
@test get_overlap(w4) == DiskArrayEngine.Overlapping()
@test get_sparsity(w4) == DiskArrayEngine.Dense()
@test w4[2] == 11:12

w5 = DiskArrayEngine.MovingWindow(200,-5,2,100)
@test get_ordering(w5) == DiskArrayEngine.Decreasing()
@test get_overlap(w5) == DiskArrayEngine.NonOverlapping()
@test get_sparsity(w5) == DiskArrayEngine.Sparse(0.4)
@test w5[2] == 195:196

w6 = DiskArrayEngine.to_window(collect(w5))
@test get_ordering(w6) == DiskArrayEngine.Decreasing()
@test get_overlap(w6) == DiskArrayEngine.NonOverlapping()
@test get_sparsity(w6) == DiskArrayEngine.Sparse(0.4032258064516129)
@test w6[2] == 195:196

end