#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <math.h>
#include <cmath>
#include <iomanip>

using namespace std;

void alignToMedian(double *daArray, int iSize) {    
    double* dpSorted = new double[iSize];
    for (int i = 0; i < iSize; ++i) dpSorted[i] = daArray[i];
    for (int i = iSize - 1; i > 0; --i) {
        for (int j = 0; j < i; ++j) {
            if (dpSorted[j] > dpSorted[j+1]) {
                double dTemp = dpSorted[j];
                dpSorted[j] = dpSorted[j+1];
                dpSorted[j+1] = dTemp;
            }
        }
    }
    double dMedian = dpSorted[(iSize/2)-1]+(dpSorted[iSize/2]-dpSorted[(iSize/2)-1])/2.0;    
    for (int i=0;i<iSize;i++) {daArray[i] = daArray[i]-dMedian;dpSorted[i] = dpSorted[i]-dMedian;}
    double dQ1 = dpSorted[(iSize/4)-1]+((dpSorted[(iSize/4)]-dpSorted[(iSize/4)-1])/2.0);
    double dQ3 = dpSorted[(iSize/4)*3-1]+((dpSorted[(iSize/4)*3+1]-dpSorted[(iSize/4)*3-1])/2.0);
    // std::cout << dpSorted[((iSize/4)*3)-2] << std::endl;
    // std::cout << dpSorted[((iSize/4)*3)-1] << std::endl;
    // // std::cout << dQ3 << std::endl;
    // std::cout << dpSorted[(iSize/4)*3] << std::endl;
    // std::cout << dpSorted[(iSize/4)*3+1] << std::endl;
    delete [] dpSorted;
    for (int i=0;i<iSize;i++) daArray[i] = daArray[i]/(dQ3-dQ1);    
}
void softmax(double *p, double *v, double b) {
	double sum = 0.0;
	double tmp[5];		
	double max_de_sum = -10000.0;
	//summing mb + mf
	for (int i=0;i<5;i++) {
		if (v[i] > max_de_sum) {
			max_de_sum = v[i];
		}
	}

	for (int i=0;i<5;i++) {
		tmp[i] = exp((v[i]-max_de_sum)*b);
		sum+=tmp[i];		
	}			

	for (int i=0;i<5;i++) {
		p[i] = tmp[i]/sum;		
	}

	for (int i=0;i<5;i++) {
		if (p[i] == 0) {
			sum = 0.0;
			for (int i=0;i<5;i++) {
				p[i]+=1e-8;
				sum+=p[i];
			}
			for (int i=0;i<5;i++) {
				p[i]/=sum;
			}			
		}
	}
}
double entropy(double *p) {
	double tmp = 0.0;
	for (int i=0;i<5;i++) {tmp+=p[i]*log2(p[i]);}
	return -tmp;
}
// void sferes_call(double * fit, const char* data_dir, double alpha_, double beta_, double noise_, double length_, double weight_, double threshold_)
void sferes_call(double * fit, const int N, const char* data_dir, double alpha_, double beta_, double noise_, double length_, double weight_, double threshold_, double sigma_, double kappa_, double shift_)
{
	///////////////////
	double max_entropy = -log2(0.2);
	// parameters
	double alpha=0.0+alpha_*(1.0-0.0);
	double beta=0.0+beta_*(100.0-0.0);
	double noise=0.0+noise_*(0.1-0.0);
	int length=1+(10-1)*length_;	
	double threshold=0.01+(max_entropy-0.01)*threshold_;
	double sigma=0.0+(20.0-0.0)*sigma_;	
	double weight=0.0+(1.0-0.0)*weight_;
	double kappa=0.0+(1.0-0.0)*kappa_;
	double shift=-20.0+(20.0+20.0)*shift_;	
	
	int nb_trials = N/4;
	int n_state = 3;
	int n_action = 5;
	int n_r = 2;	
	///////////////////
	int sari [N][4];	
	double mean_rt [15];
	double mean_model [15];	
	double values [N]; // action probabilities according to subject
	double rt [N]; // rt du model	
	double p_a_mf [n_action];
	double p_a_mb [n_action];	

	const char* _data_dir = data_dir;
	std::string file1 = _data_dir;
	std::string file2 = _data_dir;
	file1.append("sari.txt");
	file2.append("mean.txt");	
	std::ifstream data_file1(file1.c_str());
	string line;
	if (data_file1.is_open())
	{ 
		for (int i=0;i<N;i++) 
		{  
			getline (data_file1,line);			
			stringstream stream(line);
			std::vector<int> values(
     			(std::istream_iterator<int>(stream)),
     			(std::istream_iterator<int>()));
			for (int j=0;j<4;j++)
			{
				sari[i][j] = values[j];
			}
		}
	data_file1.close();	
	}
	std::ifstream data_file2(file2.c_str());	
	if (data_file2.is_open())
	{
		for (int i=0;i<15;i++) 
		{  
			getline (data_file2,line);			
			double f; istringstream(line) >> f;
			mean_rt[i] = f;
		}
	data_file2.close();	
	}		
	// for (int i=0;i<4;i++)		
	for (int i=0;i<2;i++) 
	{		
		// START BLOC //
		double p_s [length][n_state];
		double p_a_s [length][n_state][n_action];
		double p_r_as [length][n_state][n_action][n_r];				
		double p [n_state][n_action][2];		
		double values_mf [n_state][n_action];	
		double values_mb [n_action];
		double tmp [n_state][n_action][2];
		double p_ra_s [n_action][2];
		double p_a_rs [n_action][2];
		double p_r_s [2];
		double weigh[n_state];
		int n_element = 0;
		int s, a, r;		
		double Hf = 0.0;
		for (int n=0;n<n_state;n++) { 
			weigh[n] = weight;
			for (int m=0;m<n_action;m++) {
				values_mf[n][m] = 0.0;

			}
		}		
		// START TRIAL //
		for (int j=0;j<nb_trials;j++) 		
		{							
			// COMPUTE VALUE
			s = sari[j+i*nb_trials][0]-1;
			a = sari[j+i*nb_trials][1]-1;
			r = sari[j+i*nb_trials][2];							
			for (int n=0;n<n_state;n++){
				for (int m=0;m<n_action;m++) {
					p[n][m][0] = 1./30; p[n][m][1] = 1./30; 
				}}					// fill with uniform
			double entrop = max_entropy;			
			for (int n=0;n<n_action;n++){
				p_a_mb[n] = 1./n_action;
				values_mb[n] = 1./n_action;								
			}						
			int nb_inferences = 0;									
			double p_a [n_action];			
			int k = 0;
			while ( entrop > threshold && nb_inferences < n_element) {						

				// INFERENCE				
				double sum = 0.0;
				for (int n=0;n<3;n++) {
					for (int m=0;m<5;m++) {
						for (int o=0;o<2;o++) {
							p[n][m][o] += (p_s[k][n] * p_a_s[k][n][m] * p_r_as[k][n][m][o]);
							sum+=p[n][m][o];
						}
					}
				}
				for (int n=0;n<3;n++) {
					for (int m=0;m<5;m++) {
						for (int o=0;o<2;o++) {
							tmp[n][m][o] = (p[n][m][o]/sum);
						}
					}
				}
				nb_inferences+=1;
				// EVALUATION
				sum = 0.0;				
				for (int m=0;m<5;m++) {
					for (int o=0;o<2;o++) {
						p_r_s[o] = 0.0;
						sum+=tmp[s][m][o];						
					}
				}
				for (int m=0;m<5;m++) {
					for (int o=0;o<2;o++) {
						p_ra_s[m][o] = tmp[s][m][o]/sum;
						p_r_s[o]+=p_ra_s[m][o];						
					}
				}
				sum = 0.0;
				for (int m=0;m<5;m++) {
					for (int o=0;o<2;o++) {
						p_a_rs[m][o] = p_ra_s[m][o]/p_r_s[o];
					}
					values_mb[m] = p_a_rs[m][1]/p_a_rs[m][0];
					sum+=values_mb[m];
				}
				for(int m=0;m<5;m++) {
					p_a_mb[m] = values_mb[m]/sum;
				}				
				entrop = entropy(p_a_mb);
				k+=1;
				
			}	

			// FUSION	
			softmax(p_a_mf, values_mf[s], beta);
			double Hf = entropy(p_a_mf);

			double sum = 0.0;

			// for (int n=0;n<5;n++) {
			// 	std::cout << p_a_mb[n] << " ";
			// }
			// std::cout << std::endl;
			//std::cout << "W= " << weigh[s] << std::endl;

			for (int n=0;n<5;n++) {
				p_a[n] = (1.0-weigh[s])*p_a_mf[n] + weigh[s]*p_a_mb[n];
				sum+=p_a[n];
			}
			for (int n=0;n<5;n++) p_a[n]/=sum;	
			
						

			double H = entropy(p_a);
			double N = nb_inferences+1.0;

			values[j+i*nb_trials] = log(p_a[a]);
			// std::cout << i << " " << j << " " << log(p_a[a]) << std::endl;
			rt[j+i*nb_trials] = pow(log2(N), sigma) + H;
			
			// UPDATE WEIGHT
			double p_wmc;
			double p_rl;
			std::cout << r << std::endl;
			if (r == 1) {
				p_wmc = p_a_mb[a];
				p_rl = p_a_mf[a];				
			} else {
				p_wmc = 1.0 - p_a_mb[a];
				p_rl = 1.0 - p_a_mf[a];
			}			
			
			weigh[s] = (p_wmc*weigh[s])/(p_wmc*weigh[s]+p_rl*(1.0-weigh[s]));
			//MODEL FREE
			double reward;
			if (r == 0) {reward = -1.0;} else {reward = 1.0;}
			double delta = reward - values_mf[s][a];
			values_mf[s][a]+=(alpha*delta);
			// forgetting
			for (int m=0;m<5;m++) {
				if (m != a) {
					values_mf[s][m] += (1.0 - kappa)*(0.0 - values_mf[s][m]);
				}
			}			
			if (delta < shift) {						
				// UPDATE MEMORY 						
				for (int k=length-1;k>0;k--) {
					for (int n=0;n<3;n++) {
						p_s[k][n] = p_s[k-1][n]*(1.0-noise)+noise*(1.0/n_state);
						for (int m=0;m<5;m++) {
							p_a_s[k][n][m] = p_a_s[k-1][n][m]*(1.0-noise)+noise*(1.0/n_action);
							for (int o=0;o<2;o++) {
								p_r_as[k][n][m][o] = p_r_as[k-1][n][m][o]*(1.0-noise)+noise*0.5;				
							}
						}
					}
				}						
				if (n_element < length) n_element+=1;
				for (int n=0;n<3;n++) {
					p_s[0][n] = 0.0;
					for (int m=0;m<5;m++) {
						p_a_s[0][n][m] = 1./n_action;
						for (int o=0;o<2;o++) {
							p_r_as[0][n][m][o] = 0.5;
						}
					}
				}			
				p_s[0][s] = 1.0;
				for (int m=0;m<5;m++) {
					p_a_s[0][s][m] = 0.0;
				}
				p_a_s[0][s][a] = 1.0;
				p_r_as[0][s][a][(r-1)*(r-1)] = 0.0;
				p_r_as[0][s][a][r] = 1.0;
			}

		}
	}	
	
	// ALIGN TO MEDIAN
	alignToMedian(rt, N);	
	// for (int i=0;i<N;i++) std::cout << rt[i] << std::endl;
	double tmp2[15];
	for (int i=0;i<15;i++) {
		mean_model[i] = 0.0;
		tmp2[i] = 0.0;
	}

	for (int i=0;i<N;i++) {
		mean_model[sari[i][3]-1]+=rt[i];
		tmp2[sari[i][3]-1]+=1.0;				
	}	
	double error = 0.0;
	for (int i=0;i<15;i++) {
		mean_model[i]/=tmp2[i];
		error+=pow(mean_rt[i]-mean_model[i],2.0);				
	}	
	for (int i=0;i<N;i++) fit[0]+=values[i];	
	fit[1] = -error;
	
	if (std::isnan(fit[0]) || std::isinf(fit[0]) || std::isinf(fit[1]) || std::isnan(fit[1]) || fit[0] < -1e+30 || fit[1] < -1e+30) {
		fit[0]=-1e+15;
		fit[1]=-1e+15;
		return;
	}
}
