sourcecode="hyplast_core.for"
libraryname="./libplast_core.so"

all: clean libf64

libf64:
	#64bits compling
	#g++ -Wall -fPIC -c ctest1.cpp ctest2.cpp
	#g++ -shared -o libctest.so ctest1.o ctest2.o	
	#gfortran -shared -fPIC -o libftest.so ctestfor.f90
	#gfortran -fno-f2c -shared -o libftest.so ctestfor.f90
	#gfortran -fno-underscoring -m64 -fPIC -shared -o libUEL505_64.so uelvar1_l4gp.for 
	#gfortran -m64 -fPIC -shared -o libUEL505_64.so uelvar1_l4gp.for 
	#gfortran -m64 -fPIC -shared -fno-range-check -o libISTLIBPL.so multisurfaceplasticityhard.for
	gfortran -m64 -fPIC -shared -o $(libraryname) $(sourcecode)
	
libf64dbg:
	gfortran -m64 -fPIC -shared -O0 -fcheck=all -g -o $(libraryname) $(sourcecode)

libf32:
	#32bits compiling
	gfortran -fno-underscoring -m32 -fPIC -shared -o $(libraryname) $(sourcecode)
	
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
	#rm -f ctest
	rm -f *.o *.so
