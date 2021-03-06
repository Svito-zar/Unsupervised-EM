#include <iostream>
#include "lm/model.hh"  /* Language Model */
#include <fstream>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <limits> 	/* numeric limits */
#include <string>

using namespace lm::ngram;
Model model("10gram.wb.lm"); // read a 10-gram model


#define N 393  // Amount of training observations
#define D 2  // Dimensionality of a single vector 
#define C 83  // Amount of classes ( letters in current case)
#define M 2   // We will use m-gram language model

#define epsilon pow(10,-10) //accuracy of calculations

// i/o functions
void reading_file(double array[N][D],char* name);

// functions for HMM
void init_params();
double trans_prob(int prev, int next);
double emiss_prob(double x[D], int cl);
double unigram_score(int symbol);
double add_log_scores(double a, double b);

// DP backward-forward algorithm functions
void init_Q(double word[N][D]);
void init_Q_tilda(double word[N][D]);
void forward(int n, int c, double word[N][D]);
void backward(int n, int c, double word[N][D]);
void Baum_algorithm(double data[N][D]); 

// Learning functions
void EM(double data[N][D], int iterNumb);
double word_total_probability(double word[N][D]);

// Testing functions
void test_prob_norm();
void primitive_decode();
void debug();

// Global Variables
double Means[C][D]; 	// means of the classes
double Variances[D]; 	// pooled variance
double Q[N][C]; 	// backward recursion scores
double Q_tilda[N][C];   // forward recursion scores
double P[N][C]; 	// prob(c/data) for the time step n


int main()
{
	// create example observations : ien ien ien ien ...
	double x[N][D];
	// i =50, e  = 46, n = 55
	for(int n=0;n<N;n++)
	{
		for(int d=0;d<D;d++)
		{
			x[n][d] = n%83;		
			//if(n%3 ==0) x[n][d] = 50;
			//else if(n%3 ==1) x[n][d] = 46;
			//else x[n][d] = 55;
		}
	}
	x[5][0] = 35; x[5][1] = 35; // just to make training data more interesting

	// Show the example data
	std::cout<<" For debug we use the data : ";
	for(int n=0;n<N;n++)
		std::cout<<x[n][0]<<" ";
	std::cout<<"\n";

	// initialise parameters ( means and variances) and table Q
	init_params();


	// Do Expectation - Maximization
	EM(x, 5);

	// test
	//test_prob_norm();
	//debug();

	//primitive_decode();
	return 0;
}

void reading_file(double array[N][D],char* name)
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
		// read all dimensions of the data
		for(int d=0;d<D;d++)
		{
			in>>array[ind][d]; 
        	}
	}
}

// Initialize parameters of all characters : arrays of means and variances
void init_params()
{
	for(int d=0;d<D;d++)
		Variances[d] = 1;
	for(int c=0;c<C;c++)
		for(int d=0;d<D;d++)
			Means[c][d] = c+0.5;
}

// Transition probability for HMM ( ln of it)
double trans_prob(int prev, int next)
{
  	State next_state(model.BeginSentenceState()), prev_state;
	// get prev_state of a prev word
	model.Score(next_state, prev, prev_state);
	// get a score : prob(next/prev)
	double score = model.Score(prev_state, next, next_state);
	// score - log10 of a probability, but we want to have ln
	return log(pow(10,score));
}

// HMM emission probability (ln of it)
double emiss_prob(double x[D], int cl)
{
	double sum_for_exp = 0;
	double norm_fact = 1;
	// go over all dimensions
	for(int d =0; d<D;d++)
	{
		sum_for_exp += (x[d] - Means[cl][d]) * (x[d] - Means[cl][d]) / (Variances[d]*Variances[d]);
		norm_fact*=  sqrt(2*M_PI) * Variances[d];
	}
	return -0.5*sum_for_exp - log(norm_fact);
}

// ln of the unigram probability for a symbol
double unigram_score(int symbol)
{
  	State prev_state(model.BeginSentenceState()), curr_state;
	// get a score : prob(symbol)
	double score = model.Score(prev_state, symbol, curr_state);
	// score - log10 of a probability
	return log(pow(10,score));
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
    		std::cerr<< "Error : refusing to add two infinite log scores\n";
		return 0.1;
	}
}

/* 					BACKWARD - FORWARD ALGORITHM			*/

// Initialization of a table Q
void init_Q(double x[N][D])
{
/* input : training sequence */
	
	// At first time step we don't have predecessor, so we fall back to the unigram model
	for(int cl=0;cl<C;cl++) // go over all classe
	{
		Q[0][cl]  = emiss_prob(x[0],cl) + unigram_score(cl);
	}

	// All other values are initialized to -1 in order to see that it is not assigned the value yet
	for(int n=1;n<N;n++)
		for(int cl=0;cl<C;cl++) // go over all characters
			Q[n][cl]  = -1;
	
}

// Forward algorithm
void forward(int n, int curr_char, double word[N][D])
{
	if(n==0) // we can use initialization
	{
		if(Q[0][curr_char] == -1)
			std::cerr<<" Error! Q was not initialized, but forward algorithm has started";
		return;
	}
	
	// in order to avoid numerical problems
	// we assign the sum to the first term
	double sum = Q[n-1][0] + trans_prob(0, curr_char);  
	// sum over all other characters
	for(int prev_char=1;prev_char<C;prev_char++)
	{
		if(Q[n-1][prev_char] == -1)
			std::cerr<<" Error! At time step "<<n-1<<" for class "<<prev_char<<" Q was not initialized, but forward algorithm tried to use it\n";
		// We need to do summation in a logarithm space
		sum = add_log_scores(sum, Q[n-1][prev_char] + trans_prob(prev_char, curr_char));  
	}
	Q[n][curr_char] = emiss_prob(word[n],curr_char) + sum;
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
void backward(int n, int curr_char, double word[N][D])
{
	if(n==N-1) // we can use initialization
	{
		if(Q_tilda[N-1][curr_char] == -1)
			std::cerr<<" Error! Q_tilda was not initialized, but backward algorithm has started";
		return;
	}

	// in order to avoid numerical problems
	// we assign the sum to the first term
	double sum = Q_tilda[n+1][0] + trans_prob(curr_char, 0) + emiss_prob(word[n+1],0);  
	// sum over all other characters
	for(int next_char=1;next_char<C;next_char++)
	{
		//std::cout<<"Value for "<<prev_char<<" is "<<Q[n-1][prev_char]<<"\n";
		sum = add_log_scores(sum, Q_tilda[n+1][next_char] + trans_prob(curr_char, next_char) + emiss_prob(word[n+1],next_char));  
		if( n == N+2 && curr_char == 48)	
		{
			std::cout<<"For the next char "<<next_char<<" : Q prev = "<< Q_tilda[n+1][next_char]<<" trans prob ="<<trans_prob(curr_char, next_char)<<" emiss = "<<emiss_prob(word[n+1],next_char);  
			std::cout<<"\nThe sum is "<<sum<<"\n";
		}
	}
	Q_tilda[n][curr_char] = sum;
}

// Forward-backward algorithm, which will compute probabilities of each class C at time-step t given input sequence data
// It is calucaled in log-space ! 
// It will be writen into P[t][C] ( it is a global array)
void Baum_algorithm(double data[N][D])
{
	// Do forward path
	init_Q(data);
	for(int n=1;n<N;n++) // go over all timespeps
		for(int c=0;c<C;c++) // go over all characters
			forward(n,c,data);
	// Do backward path
	init_Q_tilda(data);
	for(int n=N-2;n>=0;n--) // go over all timespeps
		for(int c=0;c<C;c++) // go over all characters
			backward(n,c,data);

	// Calculate not normalized log - probabilities
	for(int n=0;n<N;n++) // go over all timespeps
		for(int c=0;c<C;c++) // go over all characters
			P[n][c] = (Q[n][c] + Q_tilda[n][c]);

	// Get a log of the total probabolity of a sequence for normalization
	double log_norm = word_total_probability(data);

	std::cout<<"Normalization factor: "<<log_norm<<"\n";
	// Normalize the probabilities
	for(int n=0;n<N;n++) // go over all timespeps
		for(int c=0;c<C;c++) // go over all characters
			P[n][c] = P[n][c] -  log_norm;
}

/* 						Learning 						*/

// Expectation Maximization algorithm
void EM(double data[N][D], int iterNumb)
{
	double sum, norm;
	for(int iter = 0; iter<iterNumb; iter++)
	{
		// Debug
		std::cout<<"Means : ";
		for(int c=0;c<C;c++)
			std::cout<<Means[c][0]<<" ";
		std::cout<<"\n";


		// first - get probabilities Pt(c / x1_N)
		Baum_algorithm(data);

		// check normalization
		test_prob_norm();
		
		// then - reestimate means
		for(int c=0;c<C;c++) // for all classes
		{
			// Calculate normalization sum
			norm = 0;
			for(int t=0;t<N;t++) // go over all time-steps
			{
				// if(c==35) std::cout<<"norm = "<<norm<<"\n";
				norm = norm + exp(P[t][c]);
			}
			if(norm < epsilon * epsilon) 
			{
				//std::cout<<"Kept the mean for class "<<c<<" because of the norm = "<<norm<<"\n";
				continue; //to avoid devition on zero
			}
			// Calculate the sum for the new mean
			for(int d=0;d<D;d++) // for all dimensions
			{
				sum = 0;
				for(int t=0;t<N;t++) // go over all time-steps
					sum = sum + exp(P[t][c]) * data[t][d];
				Means[c][d] = sum / norm;
			}
		}

		// finally reestimate gaussians
		for(int d=0;d<D;d++)
		{
			sum = 0;
			for(int c=0;c<C;c++) // for all classes
				for(int t=0;t<N;t++) // go over all time-steps
					sum = sum + exp(P[t][c]) * (data[t][d] - Means[c][d])*(data[t][d] - Means[c][d]);
			
			Variances[d] = sum / N;
		}	
		
		debug();
	}
}
			
	
// Log - probability of a given word
double word_total_probability(double word[N][D])
{
	double sum=P[N-1][0]; // initialize the sum
	for(int c=0;c<C;c++)
		sum = add_log_scores(sum,P[N-1][c]); // sum in a log space
	return sum;
}

	


/* 					Testing and Debuging						*/

// Test if our probability distrubution for classes is normalized
void test_prob_norm()
{
	double sum;
	std::cout<<"I am testing the normalization of the probability distribution ...\n";
	for(int n=0;n<N;n++)
	{
		sum = 0;
		for(int c=0;c<C;c++)
			sum = sum + exp(P[n][c]);
		if(fabs(sum - 1) > 0.0001)
		{
			std::cout<<"The probability of characters is not normalized at time-step "<<n<<". It sums up to "<<sum<<"\n";
			return;
		}
		//std::cout<<sum<<"\n";
	}
	std::cout<<"Normalization test was passed!\n";
}
		

// Was used to debug a Forward - Backward algorithm
void debug()
{
		//DEBUG
	std::cout<<"Sigma : "<<Variances[0]<<" "<<Variances[1]<<"\n";
	//for(int c=0;c<C;c++)
	//	std::cout<<"P[N-1]["<<c<<"] = "<<P[N-1][c]<<"\n";
	
	// DEBUG table Q
	// Find the highest prob at the last time-step
	double max_val=Q[N-1][0];
	int max=0;
	for(int c=1;c<C;c++)
	{
		// let's check Q now
		if(Q[N-1][c] > max_val)
		{
			max_val = Q[N-1][c];
			max = c;
		}
		// print probabilities
		//std::cout<<"Q of class "<<c<<" at the last time step is "<<Q[N-1][c]<<"\n";
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
			// let's check Q now
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
