#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <math.h>

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
	// for (int i=0;i<5;i++) std::cout << v[i] << " "; std::cout << std::endl;
	for (int i=0;i<5;i++) {
		tmp[i] = exp(v[i]*b);
		sum+=tmp[i];		
	}		
	// for (int i=0;i<5;i++) std::cout << tmp[i] << " "; std::cout << std::endl;
	for (int i=0;i<5;i++) {
		p[i] = tmp[i]/sum;		
	}
	// for (int i=0;i<5;i++) std::cout << p[i] << " "; std::cout << std::endl;
	// int ind=-1;
	// for (int i=0;i<5;i++) {
	// 	if (isnan(p[i])) {
	// 		ind = i;
	// 		break;
	// 	}
	// }
	// // std::cout << ind << std::endl;
	// if (ind!=-1) {
	// 	for (int i=0;i<5;i++) {
	// 		p[i] = 0.0001;
	// 	}
	// }
	// p[ind] = 0.9996;
	// for (int i=0;i<5;i++) std::cout << p[i] << " "; std::cout << std::endl;
	// std::cout << ind << std::endl;
	// std::cout << std::endl;
	for (int i=0;i<5;i++) {
		if (p[i] == 0) {
			sum = 0.0;
			for (int i=0;i<5;i++) {
				p[i]+=1e-4;
				sum+=p[i];
			}
			for (int i=0;i<5;i++) {
				p[i]/=sum;
			}
			return;
		}
	}	
}
double sigmoide(double Hb, double Hf, double n, double i, double t, double g) {	
	// std::cout << "Hb = "<< Hb << ", Hf = " << Hf << " n=" << n << " i=" << i << " threshold = " << t << " gamma = " << g << std::endl;
	double x = 2.0 * -log2(0.2) - Hb - Hf;
	// std::cout << pow((n-i),t) <<  std::endl;
	double tmp = 1.0/(1.0+(pow((n-i),t)*exp(-x*g)));
	// std::cout << tmp << std::endl;
	return tmp;
	// return 1.0/(1.0+((n-i)*t)*exp(-x*g));

}
void fusion(double *p_a, double *mb, double *mf, double beta) {
	double tmp[5];
	int tmp2[5];
	double sum = 0.0;
	double ninf = 0;
	// std::cout << "beta = "  << beta << std::endl;
	for (int i=0;i<5;i++) {				
		tmp[i] = exp((mb[i]+mf[i])*beta);		
		if isinf(tmp[i]) {
			tmp2[i] = 1;
			ninf += 1.0;
		} else {
			tmp2[i] = 0;
		}
		sum+=tmp[i];
	}
	
	if (ninf > 0.0) {
		for (int i=0;i<5;i++) {
			if (tmp2[i] == 1) {				
				p_a[i] = (1.0 - 0.0000001 * (5.0 - ninf))/ninf;
			}
			else {
				p_a[i] = 0.0000001;
			}
		}
	}
	else {
		for (int i=0;i<5;i++) {				
			p_a[i] = tmp[i]/sum;
		}
	}	
	for (int i=0;i<5;i++) {
		if (p_a[i] == 0) {
			sum = 0.0;
			for (int i=0;i<5;i++) {
				p_a[i]+=1e-8;
				sum+=p_a[i];
			}
			for (int i=0;i<5;i++) {
				p_a[i]/=sum;
			}
			return;
		}
	}
}
double entropy(double *p) {
	double tmp = 0.0;
	for (int i=0;i<5;i++) {tmp+=p[i]*log2(p[i]);}
	return -tmp;
}
double sum_prod(double *a, double *b, int n) {
	double tmp = 0.0;
	for (int i=0;i<n;i++) {
		tmp+=(a[i]*b[i]);
	}
	return tmp;
}
// void sferes_call(double * fit, const char* data_dir, double alpha_, double beta_, double noise_, double length_, double gain_, double threshold_, double gamma_)
void sferes_call(double * fit, const int N, const char* data_dir, double alpha_, double beta_, double noise_, double length_, double gain_, double threshold_, double gamma_, double sigma_=1.0)
{
	///////////////////
	// parameters
	double alpha=0.0+alpha_*(1.0-0.0);
	double beta=0.0+beta_*(100.0-0.0);
	double noise=0.0+noise_*(0.1-0.0);
	int length=1+(10-1)*length_;
	double gain=0.00001+(10000.0-0.00001)*gain_;
	double threshold=0.00001+(1000.0-0.00001)*threshold_;
	double sigma=0.0+(20.0-0.0)*sigma_;	
	double gamma=0.0+(100.0-0.0)*gamma_;

	// std::cout << noise << " " << length << " " << beta << " " << gain << " " << threshold << " " << alpha << " " << sigma << " " << gamma << std::endl;
	int nb_trials = N/4;
	int n_state = 3;
	int n_action = 5;
	int n_r = 2;
	double max_entropy = -log2(0.2);
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

	for (int i=0;i<4;i++)	
	// for (int i=0;i<1;i++)	
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
		int n_element = 0;
		int s, a, r;		
		double Hf = 0.0;
		for (int n=0;n<n_state;n++) { for (int m=0;m<n_action;m++) {values_mf[n][m] = 0.0;}}
		// START TRIAL //
		for (int j=0;j<nb_trials;j++) 
		{				
			// COMPUTE VALUE
			s = sari[j+i*nb_trials][0]-1;
			a = sari[j+i*nb_trials][1]-1;
			r = sari[j+i*nb_trials][2];				
			double Hb = max_entropy;
			for (int n=0;n<n_state;n++){
				for (int m=0;m<n_action;m++) {
					p[n][m][0] = 1./30; p[n][m][1] = 1./30; 
				}}					// fill with uniform
			softmax(p_a_mf, values_mf[s], gamma);
		
			double Hf = 0.0; 
			for (int n=0;n<n_action;n++){
				values_mb[n] = 1./n_action;
				// std::cout << p_a_mf[n] << " ";
				Hf-=p_a_mf[n]*log2(p_a_mf[n]);
			}
			// std::cout << std::endl;
			// std::cout << Hf << std::endl;
			int nb_inferences = 0;
			double p_decision [n_element+1];
			double p_retrieval [n_element+1];
			double p_ak [n_element+1];

			double reaction [n_element+1];
			double values_net [n_action];
			double p_a [n_action];
			p_decision[0] = sigmoide(Hb, Hf, n_element, nb_inferences, threshold, gain);
			
			p_retrieval[0] = 1.0-p_decision[0];
			fusion(p_a, values_mb, values_mf[s], beta);
			p_ak[0] = p_a[a];
			reaction[0] = entropy(p_a);
			
			for (int k=0;k<n_element;k++) {
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
				// // EVALUATION
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
					sum += values_mb[m];
				}				
				for (int m=0;m<5;m++) {
					p_a_mb[m] = values_mb[m]/sum;
				}
				Hb = entropy(p_a_mb);
				// FUSION
				fusion(p_a, values_mb, values_mf[s], beta);
				p_ak[k+1] = p_a[a];
				double N = k+2.0;
				reaction[k+1] = pow(log2(N), sigma) + entropy(p_a);
				
				// SIGMOIDE
				double pA = sigmoide(Hb, Hf, n_element, nb_inferences, threshold, gain);				

				// if (i==1 && j==15 && k==8) {
				// 	for (int m=0;m<5;m++) std::cout << p_a[m] << " "; std::cout << std::endl;					
					// for (int m=0;m<5;m++) std::cout << values_mb[m]+values_mf[s][m] << " "; std::cout << std::endl;					
					// double tmp[5];
					// double sum = 0.0;
					
					// for (int m=0;m<5;m++) {				
					// 	tmp[m] = exp((values_mb[m]+values_mf[s][m])*beta);								
					// 	sum+=tmp[m];
					// }
					// std::cout << beta << std::endl;
					// for (int m=0;m<5;m++) std::cout << tmp[m] << " "; std::cout << std::endl;					
					// for (int m=0;m<5;m++) std::cout << p_a[m] << " "; std::cout << std::endl;					
					// std::cout << k << " " << Hb << " " << Hf << " " << pA << " " << threshold << " " << gain << std::endl;
				// }
				

				// std::cout << pA << std::endl;
				p_decision[k+1] = pA*p_retrieval[k];
				p_retrieval[k+1] = (1.0-pA)*p_retrieval[k];
			}
			

			// std::cout << "mb = "; for (int k=0;k<5;k++) std::cout << values_mb[k] << " "; std::cout << std::endl;			
			// std::cout << "mf = "; for (int k=0;k<5;k++) std::cout << values_mf[s][k] << " "; std::cout << std::endl;			
			// std::cout << "p_ak = "; for (int k=0;k<n_element+1;k++) std::cout << p_ak[k] << " "; std::cout << std::endl;			
			// std::cout << "rt = "; for (int k=0;k<n_element+1;k++) std::cout << reaction[k] << " "; std::cout << std::endl;						 
			
			values[j+i*nb_trials] = log(sum_prod(p_ak, p_decision, n_element+1));
			// if (i==1 && j==15) {
			// 	std::cout << n_element << std::endl;
			// 	std::cout << "p_a(k) "; for (int k=0;k<n_element+1;k++) std::cout << p_ak[k] << " "; std::cout << std::endl;
			// 	std::cout << "p_d " << n_element+1 << " "; for (int k=0;k<n_element+1;k++) std::cout << p_decision[k] << " "; std::cout << std::endl;
			// 	std::cout << values[j+i*nb_trials] << std::endl;
			// }
			double val = sum_prod(p_ak, p_decision, n_element+1);						
			
			rt[j+i*nb_trials] = sum_prod(reaction, p_decision, n_element+1);			
			// std::cout << rt[j+i*nb_trials] << std::endl;
			// std::cout << val << std::endl;
			// std::cout << std::endl;
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
			// // MODEL FREE	
			// double reward;
			// if (r == 0) {reward = -1.0;} else {reward = 1.0;}
			// double delta = reward - values_mf[s][a];
			// values_mf[s][a]+=(alpha*delta);
			// MODEL FREE	
			double reward;
			if (r == 0) {reward = -1.0;} else {reward = 1.0;}
			double delta = reward - values_mf[s][a];
			values_mf[s][a]+=(alpha*delta);
			// if (r==1)				
			// 	values_mf[s][a]+=(alpha*delta);
			// else if (r==0)
			// 	values_mf[s][a]+=(omega*delta);
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

	// for (int i=0;i<N;i++) {
	// 	std::cout << values[i] << std::endl;
	// }

	for (int i=0;i<N;i++) fit[0]+=values[i];	
	fit[1] = -error;
	
	if (isnan(fit[0]) || isinf(fit[0]) || isinf(fit[1]) || isnan(fit[1]) || fit[0]<-10000 || fit[1]<-10000) {
		fit[0]=-1000.0;
		fit[1]=-1000.0;
		return;
	}
	else {
		fit[0]+=2000.0;
		fit[1]+=500.0;
		return ;
	}
}
