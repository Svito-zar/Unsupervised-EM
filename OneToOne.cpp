#include <iostream>
#include "lm/model.hh"  /* Language Model */
#include <fstream>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <limits> 	/* numeric limits */
#include <string>
#include <boost/foreach.hpp> // for tokenization the line
#include <boost/tokenizer.hpp>
#include <random>

using namespace lm::ngram;

Model model("10gram.wb.lm"); // read a 10-gram model


#define N 1000  // Amount of training observations
#define D 256  // Dimensionality of a single vector 
#define C 10  // Amount of classes ( numbers in current case)
#define M 1   // We will use m-gram language model
#define Range 1000 // Range of values

#define epsilon 1e-10 //accuracy of calculations

// i/o functions
void read_labels(int array[N],char* name);
int read_usps(double x[N][D], char* filename);

// functions for HMM
void init_params();
double USPS_trans_prob(int prev, int next);
double emiss_prob(double x[D], int cl);
double USPS_unigram_score(int symbol);
double add_log_scores(double a, double b);

// DP backward-forward algorithm functions
int init_Q(double word[N][D]);
void init_Q_tilda(double word[N][D]);
int forward(int n, int curr_char, double data[N][D]);
int backward(int n, int c, double word[N][D]);
int Baum_algorithm(double data[N][D]); 

// Learning functions
void EM(double data[N][D], int iterNumb);
double word_total_probability();

// Testing functions
int test_prob_norm();
void primitive_decode();
void debug(double x[N][D]);
double dist(int cl1, int cl2);

// Evaluation
double evaluate(int labels[N]);

// Global Variables
double Means[C][D]; 	// means of the classes
double Variances[D]; 	// pooled variance
double Q[N][C]; 	// backward recursion scores
double Q_tilda[N][C];   // forward recursion scores
double P[N][C]; 	// prob(c/data) for the time step n


int main()
{
	int err_flag;

	// read example observations 
	double x[N][D];
	err_flag = read_usps(x,"test.txt");

	// check if the reading was succesfull
	if(err_flag)
		return -1;

	// initialise parameters ( means and variances)
	init_params();


	//Baum_algorithm(x); 
	
	//TRAIN
	// Do Expectation - Maximization
	EM(x, 15);

	//TEST
	std::cout<<"I am evaluating...";
	int labels[N];
	read_labels(labels,"labels.txt");

	double err_rate =  evaluate(labels);
	std::cout<<"Error rate : "<<err_rate<<"\n";
	//primitive_decode();
	return 0;
}

// read a USPS data 
int read_usps(double x[N][D], char* filename)
{
	std::string line; //for reading a line
	std::ifstream myfile (filename);
	boost::char_separator<char> sep("[] ");
 	int n=0;// iterator for observations
	int d=0;//iterator for dimensions
   	if (myfile.is_open())
   	{
		std::cout<<"Reading USPS data from the file "<<filename<<" ...\n";
      		while ( myfile.good() )
       		{
           		std::getline (myfile,line);
	   		boost::tokenizer< boost::char_separator<char> > tokens(line, sep);
    	   		BOOST_FOREACH (const std::string& t, tokens) 
			{
				std::string::size_type sz;      // alias of size_t
				x[n][d] = stod (t,&sz) ;  	// get next value
				d++;
				if(d==D) // we are done with current sample
				{
					n++; // switch to next sample
					d=0;
				}
			}
      		}
       		myfile.close();
		return 0;
   	}
   	else std::cout << "Unable to open the file with USPS data\n"; 
	return 1; // if we are here - we could not open the file
}

void read_labels(int array[N],char* name)
{
        std::ifstream in(name); // open the file for reading
        if(!in)
        {
                std::cout<<"Cannot read file"<<name;
		return;
        }
	// read all data samples
        for(int ind=0; ind<N;ind++)
        {
		in>>array[ind]; 
	}
}

// Initialize parameters of all characters : arrays of means and variances
void init_params()
{
	/* initialize random seed: */
  	srand (time(NULL));
	std::default_random_engine generator;
  	std::uniform_real_distribution<double> distribution(0.0,1000);
	for(int d=0;d<D;d++)
		Variances[d] = 80;
	for(int c=0;c<C;c++)
		for(int d=0;d<D;d++)
			Means[c][d] = distribution(generator); // generate random number
}


// HMM emission probability (ln of it)
double emiss_prob(double x[D], int cl)
{
	double sum_for_exp = 0;
	double norm_fact = 0;
	// go over all dimensions
	for(int d =0; d<D;d++)
	{
		// Ignore not relevant features
		if(fabs(Variances[d]) < epsilon)
			continue;
		sum_for_exp += (x[d] - Means[cl][d]) * (x[d] - Means[cl][d]) / (Variances[d]*Variances[d]);
		norm_fact+=  log(sqrt(2*M_PI) * Variances[d]);
		// Check for numerical problems
		if(std::isnan(sum_for_exp))
			std::cerr<<"Class : "<<cl<<" "<<" after dimension "<<d<<" : Difference for exponent of Gaussian emmision prob. is NAN ! Variances[d] = "<<Variances[d]<<" \n";
		if(std::isinf(sum_for_exp))
			std::cerr<<"Class : "<<cl<<" "<<" after dimension "<<d<<" : Difference for exponent of Gaussian emmision prob. is -inf !\n Variances[d] = "<<Variances[d]<<" \n";
		if(std::isnan(norm_fact))
			std::cerr<<"Class : "<<cl<<" "<<" after dimension "<<d<<" : Normalization factor of Gaussian emmision prob. is NAN ! Variances[d] = "<<Variances[d]<<" \n";
		if(std::isinf(norm_fact)) 
			std::cerr<<"Class : "<<cl<<" " <<" after dimension "<<d<<" :Normalization factor of Gaussian emmision prob. is -inf ! Variances[d] = "<<Variances[d]<<" \n";
	}
	// check numerical problems 
	if(std::isinf(-0.5*sum_for_exp - norm_fact)) 
		std::cerr<<"Class : "<<cl<<" has Gaussian emmision prob. = inf !  Sum for exponen = "<<sum_for_exp<<" normalization factor = "<<norm_fact<<"\n";
	if((-0.5*sum_for_exp - norm_fact) > 0) 
		std::cerr<<"Class : "<<cl<<" has Gaussian emmision prob. >0 !  Sum for exponen = "<<sum_for_exp<<" normalization factor = "<<norm_fact<<"\n";
	
	return -0.5*sum_for_exp- norm_fact;
}

// Transition probability for USPS distribution - simply unigram probabilities
// Current distribution : p(0) = 0.4, p(1) = 0.3, p(2) = 0.2, p(3) = 0.1
double USPS_trans_prob(int prev, int next)
{
	USPS_unigram_score(next);
}

// Unigram probability for USPS distribution.
// Current distribution : p(0) = 0.4, p(1) = 0.3, p(2) = 0.2, p(3) = 0.1
double USPS_unigram_score(int number)
{
  	if(number == 0)
		return log(0.4);
	else if(number == 1) 
		return log(0.3);
	else if( number == 2)
		return log(0.2);
	     else if(number ==3)
			return log(0.1);
		  else
			return -5000; // instead of -inf ( not to create numerical problems )
}

// Adding two scores in a log-space (when a and b - log probabilities)
double add_log_scores(double a, double b) {
	if(std::isfinite(a) || std::isfinite(b))
  	{
  		if (a > b) {
    			return a+log1p(exp(b-a));
  		} else {
    			return b+log1p(exp(a-b));
  		}
	}
	else
	{
    		std::cerr<< "Numerical Error : refusing to add two infinite log scores\n";
		return 10000; // just a fake value in order to be able to check
	}
}

/* 					BACKWARD - FORWARD ALGORITHM			*/

// Initialization of a table Q
int init_Q(double x[N][D])
{
/* input : training sequence */
	
	// At first time step we don't have predecessor, so we fall back to the unigram model
	for(int cl=0;cl<C;cl++) // go over all classe
	{
		Q[0][cl]  = emiss_prob(x[0],cl) + USPS_unigram_score(cl);
		// check if the value make sence
		if(Q[0][cl] > 0)
		{
			std::cout<<"Error during initialization of a Q table : positive log-probability for class "<<cl<<"\n";
			std::cout<<"Emmision prob was "<<emiss_prob(x[0],cl)<<" transition was "<<USPS_unigram_score(cl)<<"\n";
			return 0;
		}
		//std::cout<<"During initialization emission prob. of class "<<cl<<" was : "<< emiss_prob(x[0],cl)<<"\n";
	}

	// All other values are initialized to -1 in order to see that it is not assigned the value yet
	for(int n=1;n<N;n++)
		for(int cl=0;cl<C;cl++) // go over all characters
			Q[n][cl]  = -1;
			
	//std::cout<<"For x[0] at t=0 : Emmision prob of class 3"<<emiss_prob(x[0],3)<<" transition was "<<USPS_unigram_score(3)<<"\n";
	
	return 1;
}

// Forward algorithm
int forward(int n, int curr_char, double data[N][D])
{
// tries to calculate Q[n][curr_char] for our data 
	if(n==0) // we can use initialization
	{
		if(Q[0][curr_char] == -1)
			std::cerr<<" Error! Q was not initialized, but forward algorithm has started";
		return 0;
	}
	
	// in order to avoid numerical problems
	// we assign the sum to the first term
	double sum = Q[n-1][0] + USPS_trans_prob(0, curr_char);  
	// sum over all other characters
	for(int prev_char=1;prev_char<C;prev_char++)
	{
		if(Q[n-1][prev_char] == -1)
			std::cerr<<" Error! At time step "<<n-1<<" for class "<<prev_char<<" Q was not initialized, but forward algorithm tried to use it\n";
		// We need to do summation in a logarithm space
		sum = add_log_scores(sum, Q[n-1][prev_char] + USPS_trans_prob(prev_char, curr_char)); 
		// Check if we had numerical issues :
		if(sum == 10000)
		{
			std::cerr<<"Error occured while calculating forward part of Baum-Weill recursion for n = "<<n<<" curr_char = "<<curr_char<<" prev_char = "<<prev_char<<".\n";
			std::cerr<<"Q["<<n-1<<"]["<<prev_char<<"] = "<<Q[n-1][prev_char]<<" transition prob was "<< USPS_trans_prob(prev_char, curr_char)<<"\n";
			return 0; // return error code
		}
	}
	Q[n][curr_char] = emiss_prob(data[n],curr_char) + sum;

	// check if the value make sence
	if(Q[n][curr_char] > 0)
	{
		std::cout<<"Error during forward path : positive log-probability at time step "<<n<<" for class "<<curr_char<<"\n";
		std::cout<<"Sum was "<<sum<<"\n";
		return 0;
	}
	
	// if we are here - we had no errors
	return 1;
}

// Initialization of a table Q_tilda
void init_Q_tilda(double x[N][D])
{
/* input : training sequence */
	
	for(int cl=0;cl<C;cl++) // go over all classe
		Q_tilda[N-1][cl]  = pow(10, -300); // instead of 0 to avoid numerical problems

	// All other values are initialized to -1 in order to see that it is not assigned the value yet
	for(int n=0;n<N-1;n++)
		for(int cl=0;cl<C;cl++) // go over all characters
			Q_tilda[n][cl]  = -1;
	
}

// Backward algorithm
int backward(int n, int curr_char, double data[N][D])
{
	if(n==N-1) // we can use initialization
	{
		if(Q_tilda[N-1][curr_char] == -1)
			std::cerr<<" Error! Q_tilda was not initialized, but backward algorithm has started";
		return 0;
	}
	// in order to avoid numerical problems
	// we assign the sum to the first term
	double sum = Q_tilda[n+1][0] + USPS_trans_prob(curr_char, 0) + emiss_prob(data[n+1],0);  
	// Check if we had numerical issues :
	if(sum == 10000)
	{
		std::cerr<<"Error occured while staring backward part of Baum-Weill recursion for n = "<<n<<" curr_char = "<<curr_char<<".\n";
		return 0; // return error code
	}
	// sum over all other characters
	for(int next_char=1;next_char<C;next_char++)
	{
		//std::cout<<"Value for "<<prev_char<<" is "<<Q[n-1][prev_char]<<"\n";
		sum = add_log_scores(sum, Q_tilda[n+1][next_char] + USPS_trans_prob(curr_char, next_char) + emiss_prob(data[n+1],next_char));  
		if( n == N+2 && curr_char == 48)	
		{
			std::cout<<"For the next char "<<next_char<<" : Q prev = "<< Q_tilda[n+1][next_char]<<" trans prob ="<<USPS_trans_prob(curr_char, next_char)<<" emiss = "<<emiss_prob(data[n+1],next_char);  
			std::cout<<"\nThe sum is "<<sum<<"\n";
		}
		// Check if we had numerical issues :
		if(sum == 10000)
		{
			std::cerr<<"Error occured while doing backward part of Baum-Weill recursion for n = "<<n<<" curr char = "<<curr_char<<" next char = "<<next_char<<".\n";
			return 0; // return error code
		}
	}
	Q_tilda[n][curr_char] = sum;
	return 1;
}

// Forward-backward algorithm, which will compute probabilities of each class C at time-step t given input sequence data
// It is calucaled in log-space ! 
// It will be writen into P[t][C] ( it is a global array)
int Baum_algorithm(double data[N][D])
{
	int err_flag;

	// Initialise Q table
	err_flag = init_Q(data);

	// check numerical issues
	if(err_flag == 0)  
		return 0;

	// Do forward path
	for(int n=1;n<N;n++){ // go over all timespeps
		for(int c=0;c<C;c++){ // go over all characters
			err_flag = forward(n,c,data);
			if(err_flag == 0)  // check numerical issues
				return 0;
		}
	}

	// Do backward path
	init_Q_tilda(data);
	for(int n=N-2;n>=0;n--){ // go over all timespeps
		for(int c=0;c<C;c++){ // go over all characters
			backward(n,c,data);
			if(err_flag == 0)  // check numerical issues
				return 0;
		}
	}
	// Calculate not normalized log - probabilities
	for(int n=0;n<N;n++) // go over all timespeps
		for(int c=0;c<C;c++) // go over all characters
			P[n][c] = (Q[n][c] + Q_tilda[n][c]);

	// Get a log of the total probabolity of a sequence for normalization
	double log_norm = word_total_probability();

	std::cout<<"Normalization factor: "<<log_norm<<"\n";
	// Normalize the probabilities
	for(int n=0;n<N;n++) // go over all timespeps
		for(int c=0;c<C;c++) // go over all characters
			P[n][c] = P[n][c] -  log_norm;
	return 1;
}

/* 						Learning 						*/

// Expectation Maximization algorithm
void EM(double data[N][D], int iterNumb)
{
	std::cout<<"I do EM ...\n";
	double sum, norm;
	int err_flag;
	for(int iter = 0; iter<iterNumb; iter++)
	{
		std::cout<<iter<<" iteration.\n";
		// Debug
		/*
		std::cout<<"Rounded Means of class 2 : \n";
		for(int d2=0;d2<16;d2++) // second dimension of the image
		{
			for(int d1=0;d1<16;d1++)
			{
				if(Means[2][d1+d2*16] > 500)
					std::cout<<1<<" ";
				else std::cout<<0<<" ";
				//std::cout<<Means[3][d1+d2*16]<<" ";
			}
			std::cout<<"\n";
		}*/
		
		std::cout<<"Rounded Means of class 3 : \n";
		for(int d2=0;d2<16;d2++) // second dimension of the image
		{
			for(int d1=0;d1<16;d1++)
			{
				if(Means[3][d1+d2*16] > 500)
					std::cout<<1<<" ";
				else std::cout<<0<<" ";
				//std::cout<<Means[3][d1+d2*16]<<" ";
			}
			std::cout<<"\n";
		}

		// first - get probabilities Pt(c / x1_N)
		err_flag = Baum_algorithm(data);

		// check if we had numerical problems
		if(err_flag == 0)
		{
			std::cout<<" Have got numerical problems during "<<iter+1<<" iteration of EM algorithm!\n";
			return;
		}

		//debug(data);

		// check normalization
		err_flag = test_prob_norm();
		if(err_flag == 0)
		{
			std::cout<<" Have got non-normalized probability distributrion during "<<iter+1<<" iteration of EM algorithm!\n";
			return;
		}

		
		// then - reestimate means
		for(int c=0;c<C;c++) // for all classes
		{
			// Calculate normalization sum
			norm = 0;
			for(int t=0;t<N;t++) // go over all time-steps
			{
				/*if(c==2)
					if(t<10)
						if(exp(P[t][c]) > 1) 
							 std::cout<<"Prob["<<t<<"][2] = "<<exp(P[t][c])<<"\n";*/
				norm = norm + exp(P[t][c]);
			}
			if(norm < epsilon) 
			{
				//std::cout<<"Kept the mean for class "<<c<<" because of the probability sum = "<<norm<<"\n";
				continue; //to avoid devition on zero
			}
			//std::cout<<"In EM norm for class "<<c<<" = "<<norm<<"\n";
			// Calculate the sum for the new mean
			for(int d=0;d<D;d++) // for all dimensions
			{
				sum = 0;
				for(int t=0;t<N;t++) // go over all time-steps
					sum = sum + exp(P[t][c]) * data[t][d];
				Means[c][d] = sum / norm;
			}
		}

		// Reestimate gaussians
		
		for(int d=0;d<D;d++)
		{
			sum = 0;
			for(int c=0;c<C;c++) // for all classes
				for(int t=0;t<N;t++) // go over all time-steps
					sum = sum + exp(P[t][c]) * (data[t][d] - Means[c][d])*(data[t][d] - Means[c][d]);
			Variances[d] = sqrt(sum/N);
			// Regularization
			if(Variances[d] < 40)
				Variances[d] = 40;
		}	

		// Check if a few classes are two close to each other
		if(iter%4 != 3)
			continue; // do checking once in 4 time steps
		double diff; // difference between means
		for(int cl1=0;cl1<C;cl1++)
		{
			for(int cl2=cl1+1;cl2<C;cl2++)
			{
				diff = dist(cl1,cl2);
				if(diff < 0.015) // if they are too close
				{
					std::cout<<"Classes "<<cl1<<" and "<<cl2<<" collapsed!\n";
					// Reset one of them to random values
	  				srand (time(NULL));
					std::default_random_engine generator;
  					std::uniform_real_distribution<double> distribution(0.0,1000);
					for(int d=0;d<D;d++)
						Means[cl2][d] = distribution(generator); // generate random number
					// Reset variances
					for(int d=0;d<D;d++)
						Variances[d] = 120;
				}
			}
		}
	}
}
			
	
// Log - probability of a given word
double word_total_probability()
{
	double sum=P[N-1][0]; // initialize the sum
	for(int c=1;c<C;c++)
		sum = add_log_scores(sum,P[N-1][c]); // sum in a log space
	return sum;
}

	


/* 					Testing and Debuging						*/

// Test if our probability distrubution for classes is normalized
int test_prob_norm()
{
	double sum, prob;
	//std::cout<<"I am testing the normalization of the probability distribution ...\n";
	for(int n=N-1;n<N;n++)
	{
		sum =P[n][0]; // initialize the sum
		for(int c=1;c<C;c++)
			sum = add_log_scores(sum,P[n][c]); // sum in a log space
		prob = exp(sum); //convert back to the probability
		if(fabs(prob - 1) > 0.001)
		{
			std::cout<<"The probability of characters is not normalized at time-step "<<n<<". It sums up to "<<prob<<"\n";
			return 0;
		}
		//std::cout<<sum<<"\n";
	}
	//std::cout<<"Normalization test was passed!\n";
	return 1;
}
		

// Was used to debug a Forward - Backward algorithm
void debug(double x[N][D])
{
	std::cout<<"Rounded Means of class 1 : \n";
	for(int d2=0;d2<16;d2++) // second dimension of the image
	{
		for(int d1=0;d1<16;d1++)
		{
			if(Means[1][d1+d2*16] > 500)
				std::cout<<1<<" ";
			else std::cout<<0<<" ";
			//std::cout<<Means[3][d1+d2*16]<<" ";
		}
		std::cout<<"\n";
	}
	std::cout<<"Rounded Means of class 0 : \n";
	for(int d2=0;d2<16;d2++) // second dimension of the image
	{
		for(int d1=0;d1<16;d1++)
		{
			if(Means[0][d1+d2*16] > 500)
				std::cout<<1<<" ";
			else std::cout<<0<<" ";
			//std::cout<<Means[3][d1+d2*16]<<" ";
		}
		std::cout<<"\n";
	}

	std::cout<<"Rounded Observation x[2] : \n";
	for(int d2=0;d2<16;d2++) // second dimension of the image
	{
		for(int d1=0;d1<16;d1++)
		{
			if(x[2][d1+d2*16] > 500)
				std::cout<<1<<" ";
			else std::cout<<0<<" ";
			//std::cout<<Means[3][d1+d2*16]<<" ";
		}
		std::cout<<"\n";
	}
	std::cout<<"Emiss prob for time step 2:\n";
	for(int cl=0;cl<C;cl++)
		std::cout<<"Class "<<cl<<" = "<<emiss_prob(x[2], cl)<<"\n";
		
	//DEBUG
	std::cout<<"Sigma : ";
	for(int d=0;d<20;d++)
		std::cout<<Variances[d]<<" ";
	std::cout<<"\n";
	/*
	std::cout<<"\nTime step : 0 \n";
	int n=0;
	for(int c=0;c<C;c++)
		std::cout<<"Q["<<n<<"]["<<c<<"] = "<<Q[n][c]<<"\n";
	for(int c=0;c<C;c++)
		std::cout<<"Q_tilda["<<n<<"]["<<c<<"] = "<<Q_tilda[n][c]<<"\n";
	for(int c=0;c<C;c++)
		std::cout<<"P["<<n<<"]["<<c<<"] = "<<P[n][c]<<"\n";
	*/

	// DEBUG table Q
	// Find the highest prob at the first time-step
	double max_val=Q[0][0];
	int max=0;
	for(int c=0;c<C;c++)
	{
		// let's check Q now
		if(Q[N-1][c] > max_val)
		{
			max_val = Q[N-1][c];
			max = c;
		}
		// print probabilities
		//std::cout<<"Q of class "<<c<<" at the first time step is "<<Q[0][c]<<"\n";
	}
	//std::cout<<"Q gives the most probable class for the last timestep : "<<max<<" with the probability "<<max_val<<".\n";

	// DEBUG table Q_tilda
	// Find the highest prob at the first time-step
	int time_step = 1;
	max_val = Q_tilda[time_step][0];
	max=0;
	for(int c=1;c<C;c++)
	{
		// let's check Q_tilda now
		if(Q_tilda[time_step][c] > max_val) 
		{
			max_val = Q_tilda[time_step][c];
			max = c;
		}
		// print probabilities
		//std::cout<<c<<" Q_tilda value for time step = "<<time_step<<" is "<<Q_tilda[time_step][c]<<"\n";
	}
	//std::cout<<"Q_tilda gives the most probable class for the time step "<<time_step<<" : "<<max<<" with the probability "<<max_val<<".\n";
}

void primitive_decode()
{
	double max_val; 
	int max;
	std::cout<<"Sequence of classes with the maximal score ";
	for(int n=0;n<N;n++) // for each time step
	{
		max_val=P[n][0];
		max = 0;
		for(int c=1;c<C;c++)
		{
			if(P[n][c] > max_val)
			{
				max_val = P[n][c];
				max = c;
			}
		}
		// print the most probable class
		std::cout<<max<<" ";
	}
	std::cout<<"\n";
}

// will return error rate in percents
double evaluate(int labels[N])
{
	double err_rate=0;
	double max_val; 
	int max;
	int confusion[C][C];
	// initialize confusion matrix
	for(int c1=0;c1<C;c1++)
		for(int c2=0;c2<C;c2++)
			confusion[c1][c2] = 0;
	// calculate error rate	
	for(int n=0;n<N;n++) // for each time step
	{
		max_val=P[n][0];
		max = 0;
		for(int c=1;c<C;c++)
		{
			if(P[n][c] > max_val)
			{
				max_val = P[n][c];
				max = c;
			}
		}
		if(max != labels[n])
			err_rate++;
		confusion[labels[n]][max] ++; 
	}
	// print confusion matrix
	std::cout<<"Confusion matrix : \n";
	for(int c1=0;c1<C;c1++)
	{
		for(int c2=0;c2<C;c2++)
			std::cout<<confusion[c1][c2]<<" ";
		std::cout<<"\n";
	}
	return err_rate * 100.0 / N;
}

// Will calculate the L2-distance between means, normalised w.r.t. the range of values
double dist(int cl1, int cl2)
{
	double sum=0;
	for(int d=0;d<D;d++)
	{
		sum+=(Means[cl1][d] - Means[cl2][d])*(Means[cl1][d] - Means[cl2][d]);
	}
	sum = sqrt(sum) / (D * Range);
	return sum;
}
