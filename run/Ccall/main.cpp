#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <map>
// #include "call_5_3.cpp"
#include "call_4_1.cpp"
// #include "call_4_2.cpp"

using namespace std;



int main () {	

	double fit [2] = {0.0, 0.0};
	int N = 192; // meg	
	fit[0] = 0.0; fit[1] = 0.0;	

	sferes_call(fit, N, "data_meg/S7/", 1, 0.0471543, 0.6499, 0.942551, 0.0437486, 0.144935, 0.00997335, 0.0964368, 1, 0.480139 );

	// std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}