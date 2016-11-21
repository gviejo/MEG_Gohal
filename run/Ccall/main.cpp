#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <map>
// #include "call_5_3.cpp"
#include "call_4_2.cpp"
// #include "call_4_2.cpp"

using namespace std;



int main () {	

	double fit [2] = {0.0, 0.0};
	int N = 192; // meg	
	fit[0] = 0.0; fit[1] = 0.0;	

	sferes_call(fit, N, "data_meg/S9/", 0.700298, 0.0357003, 1, 0.563057, 0, 0.955392, 0.0768195, 0.741708, 0.4771);

	std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}