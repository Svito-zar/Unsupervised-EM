#include <iostream>
#include "lm/model.hh"  /* Language Model */
#include <fstream>
#include <cmath>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <limits> 	/* numeric limits */
#include <string>

using namespace lm::ngram;
Model model("10gram.wb.lm"); // read a 10-gram model


#define N 10  // Amount of training observations
#define D 2  // Dimensionality of a single vector 
#define C 83  // Amount of classes ( letters in current case)
#define M 8   // We will use m-gram language model

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
double word_total_probability(double* word, int length);

// Global Variables
double Means[C][D]; 	// means of the classes
double Variances[D]; 	// pooled variance
double Q[N][C]; 	// backward recursion scores
double Q_tilda[N][C];   // forward recursion scores
double P[N][C]; 	// prob(c/data) for the time step n


int main()
{
	// create example observations
	double x[N][D];
	for(int n=0;n<N;n++)
		for(int d=0;d<D;d++)
			x[n][d] = n; 

	// initialise parameters ( means and variances) and table Q
	init_params();

	// run forward part of the algorithm
	Baum_algorithm(x);

	// DEBUG
	for(int c=0;c<C;c++)
		std::cout<<c<<" value is "<<P[9][c]<<"\n";
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
			Means[c][d] = c;//0.1;
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
	return -0.5*sum_for_exp + log(norm_fact);
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
		//std::cout<<"Q[0]["<<cl<<"] = "<<Q[0][cl]<<"\n";
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

	double sum = 0; 
	// go over all possible characters
	for(int prev_char=0;prev_char<C;prev_char++)
	{
		//std::cout<<"Value for "<<prev_char<<" is "<<Q[n-1][prev_char]<<"\n";
		if(Q[n-1][prev_char] == -1)
			std::cerr<<" Error! At time step "<<n-1<<" for class "<<prev_char<<" Q was not initialized, but forward algorithm tried to use it\n";
		// We need to do summation in a logarithm space
		sum = add_log_scores(sum, Q[n-1][prev_char] + trans_prob(prev_char, curr_char));  
		//std::cout<<"Transition between "<<prev_char<<" and "<<curr_char<<" is "<<trans_prob(prev_char+4, curr_char+4)<<"\n";
	}
	//std::cout<<"The sum is : "<<sum<<"\n";
	// I am adding 4 to the charecter value, since index(a) = 4
	Q[n][curr_char] = emiss_prob(word[n],curr_char) + sum;
	//std::cout<<"Q["<<n<<"]["<<curr_char<<"] = "<<Q[n][curr_char]<<"\n";
}

// Initialization of a table Q_tilda
void init_Q_tilda(double x[N][D])
{
/* input : training sequence */
	
	// At first time step we don't have predecessor, so we fall back to the unigram model
	for(int cl=0;cl<C;cl++) // go over all classe
		Q_tilda[N-1][cl]  = emiss_prob(x[N-1],cl) + unigram_score(cl);

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
		if(Q[N-1][curr_char] == -1)
			std::cerr<<" Error! Q_tilda was not initialized, but backward algorithm has started";
		return;
	}

	double sum = 0; 
	// go over all possible characters
	for(int next_char=0;next_char<C;next_char++)
	{
		//std::cout<<"Value for "<<prev_char<<" is "<<Q[n-1][prev_char]<<"\n";
		sum = add_log_scores(sum, Q[n+1][next_char] + trans_prob(curr_char, next_char) + emiss_prob(word[n+1],next_char));  
		// std::cout<<"Transition between "<<prev_char<<" and "<<curr_char<<" is "<<trans_prob(prev_char+4, curr_char+4)<<"\n";
	}
	// I am adding 4 to the charecter value, since index(a) = 4
	Q[n][curr_char] = sum;
}

void Baum_algorithm(double data[N][D])
{
	// Do forward path
	init_Q(data);
	for(int n=1;n<N;n++) // go over all timespeps
		for(int c=0;c<C;c++) // go over all characters
			forward(n,c,data);
	//std::cout<<"I finished forward path\n";
	// Do backward path
	init_Q_tilda(data);
	for(int n=N-2;n>=0;n--) // go over all timespeps
		for(int c=0;c<C;c++) // go over all characters
			backward(n,c,data);
	//std::cout<<" Backward!\n";

	// Calculate the probabilities
	for(int n=0;n<N;n++) // go over all timespeps
		for(int c=0;c<C;c++) // go over all characters
			P[n][c] = Q[n][c] * Q_tilda[n][c];

}
// Probability of a given word
double word_total_probability(double* word, int length)
{
	double best=0;
	// find the best value of the probability for a cymbol at the last position
	return best;
}
