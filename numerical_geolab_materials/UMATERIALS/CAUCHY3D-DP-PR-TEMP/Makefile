sourcecode_core="../CORE/V1_2019-01-06/hyplast_core.f90"
sourcecode_material_def="hyplast_material_Cauchy3D-DP-PR-TEMP.for"
sourcecode_caller="hyplast_Cauchy3D-DP-PR-TEMP.for"
libraryname="./libplast_Cauchy3D-DP-PR-TEMP.so"

release: clean libf64
debug: clean libf64dbg

libf64:
	gfortran -m64 -fPIC -shared -Ofast -o $(libraryname) $(sourcecode_core) $(sourcecode_material_def) $(sourcecode_caller)
	
libf64dbg:
	gfortran -m64 -fPIC -shared -O0 -fcheck=all -g -o $(libraryname) $(sourcecode_core) $(sourcecode_material_def) $(sourcecode_caller)

libf32:
	#32bits compiling
	gfortran -fno-underscoring -m32 -fPIC -shared -o $(libraryname) $(sourcecode_core) $(sourcecode_material_def) $(sourcecode_caller)
	
libc64:
	gcc -Wall -fPIC -c testso.c
	gcc -shared -o libtestso.so testso.o
	
cython:
	cython3 --embed -o pytemp.c pytest.py
	gcc -Os -I /usr/include/python3.5m -o pyexec pytemp.c -lpython3.5m -lpthread -lm -lutil -ldl
	
	
bin:
	#g++ ctest.c -L./lib -lmylibtest -o ctest
	g++ ctest.c -L./lib -lmylibtest -o ctest

exe:
	LD_LIBRARY_PATH=/home/IST/myprojects/outcalls/lib ./ctest

pyth:
	#gcc -c -I/usr/include/python2.7/ -o test.o test.c
	#ar rcs libtest.a test.o
	
	gfortran -m64 -c -fno-range-check -o test.o multisurfaceplasticityhard.for
	ar rcs libISTPL.a test.o

clean:
	rm -f *.o *.so
