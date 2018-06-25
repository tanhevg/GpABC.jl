 open("REQUIRE") do f
     for s in eachline(f)
         if s[1:5] != "julia"
             Pkg.add(s)
         end
     end
 end
