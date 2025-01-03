@testset "Output buffer collections" begin
  using DiskArrayEngine
  using Test
  using DiskArrayEngine: ProductArray, LoopWindows, to_window, generate_outbuffer_collection, extract_outbuffer, put_buffer

  lr = ProductArray(([1:2,3:3,4:9,10:15,16:20],[i:i for i in 1:3],[i:i+1 for i in 1:2:4]))
  outspecs = create_outwindows((7,4),dimsmap=(1,3),windows=([1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7],1:4))
  
  @test DiskArrayEngine.bufferrepeat(lr[1,2,2],lr,outspecs.lw) == 6
  @test DiskArrayEngine.bufferrepeat(lr[2,2,2],lr,outspecs.lw) == 6
  @test DiskArrayEngine.bufferrepeat(lr[3,2,2],lr,outspecs.lw) == 3
  
  
  buftype = Int32
  init = Int32(0)
  obc = generate_outbuffer_collection(outspecs,buftype,lr,identity)
  bufview1 = extract_outbuffer(lr[1,1,1],outspecs,init,buftype,obc)
  @test length(obc.buffers) == 1
  k1 = DiskArrayEngine.BufferIndex((1:1,1:2))
  @test haskey(obc.buffers,k1)
  ob = obc.buffers[k1]
  @test ob isa DiskArrayEngine.OutArrayBuffer
  @test ob.nwritten[] == 1
  @test ob.ntot == 6
  @test DiskArrayEngine.getloopinds(ob) == (1,3)
  @test ob.offsets == (0,0)
  @test ob.a == zeros(Int32,2,2)
  @test bufview1 === ob
  bufview2 = extract_outbuffer(lr[2,1,1],outspecs,init,buftype,obc)
  ob2 = obc.buffers[k1]
  @test ob2.nwritten[] == 2
  @test ob.ntot == 6
  @test ob2 === ob
  @test length(obc.buffers) == 1
  @test bufview2 === ob
  
  outar = zeros(Int32,7,4)
  ob.a .= 1
  @test put_buffer(lr[1,1,1], bufview1, outar, nothing) == false
  @test put_buffer(lr[2,1,1], bufview1, outar, nothing) == false
  ob.nwritten[] = 6
  @test put_buffer(lr[2,3,1], bufview1, outar, nothing) == true
  DiskArrayEngine.clean_aggregator(obc)
  @test length(obc.buffers) == 0
  @test outar[1,:] == [1,1,0,0]
  @test all(iszero,outar[2:end,:])
  
  bufview2 = extract_outbuffer(lr[3,1,1],outspecs,init,buftype,obc)
  k2 = DiskArrayEngine.BufferIndex((2:3,1:2))
  ob = obc.buffers[k2]
  @test ob.nwritten[]==1
  @test ob.ntot==3
  ob.a .= 2
  ob.nwritten[] = 4
  @test_throws Exception put_buffer(lr[3,3,1], bufview2, outar, nothing)
  ob.nwritten[] = 3
  @test put_buffer(lr[3,3,1], bufview2, outar, nothing) == true
  @test all(==(2),outar[2:3,1:2])
end