#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <map>
// #include "call_5_3.cpp"
#include "call_5_2.cpp"
// #include "call_4_2.cpp"

using namespace std;



int main () {	

	double fit [2] = {0.0, 0.0};
	int N = 192; // meg	
	fit[0] = 0.0; fit[1] = 0.0;	

	sferes_call(fit, N, "data_meg/S12/", 0.0238764, 0.302694, 0.990068, 0.740149, 0.0251457, 0.911783, 0.132816, 0.992032, 0.475574, 0.520843);

	std::cout << fit[0] << " " << fit[1] << std::endl;  
   	return 0;
}