#include <iostream>
#include <iomanip>
//#include "lm/model.hh" 
#include "events.hh"     /* Language Model */
#include <sstream>
#include <fstream>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <limits> 	/* numeric limits */
#include <string>
#include <boost/foreach.hpp> // for tokenization the line
#include <boost/tokenizer.hpp>
#include <random>

//using namespace lm::ngram;

//Model model("10gram.wb.lm"); // read a 10-gram LM model

#define S 30  	   // Amount of sentences
#define T 200      // Amount of training observations - 10^4 - 10^5
#define D 784      // Dimensionality of a single vector 
#define C 10       // Amount of classes ( numbers in current case)
#define M 2   	   // We will use m-gram language model
#define Range 256  // Range of values
#define L 4        // Maximal amount of mixure densities
#define ITER 6	   // Amount of EM iteration

#define PictureAmount 2000

#define epsilon 1e-9 //accuracy of calculations

// i/o functions
bool read_mnist(std::vector<std::vector<std::vector<double> > >  MNIST, std::string folder);

// Initializations
void init_params();
double generate_LM(unsigned seed);
double sample_sequence(unsigned seed, int index);
void sample_MNIST(std::vector<std::vector<std::vector<double> > >  MNIST, int sentence_ind, std::vector<std::vector<double> > & x);

// functions for HMM
double MNIST_trans_prob(int prev, int next);
double mix_emiss_prob(std::vector<double> x, int cl,int m);
double emiss_prob(std::vector<double> x, int cl);
double MNIST_unigram_score(int symbol);
double add_log_scores(double a, double b);
double emiss_prob(std::vector<double> x, int cl);
double mix_proportion(std::vector<double> x, int cl, int l);

// DP backward-forward algorithm functions
int init_Q(std::vector<std::vector<double> >  x);
void init_Q_tilda(std::vector<std::vector<double> >  x);
int forward(int n, int curr_char, std::vector<std::vector<double> >  x);
int backward(int n, int c, std::vector<std::vector<double> >  x);
int Baum_algorithm(std::vector<std::vector<double> >  x); 
double word_total_probability();

// Learning functions
void EM();
void addToSum(std::vector<std::vector<double> >  x, std::vector<std::vector<std::vector<double> > > & sum_means, std::vector<std::vector<double> >  & sum_gama_cl, std::vector<double> & sum_gama_c, std::vector<double>  sum_variances);

// Testing functions
int test_prob_norm();
void primitive_decode();
void debug(std::vector<std::vector<double> >  x);
double dist(int cl1, int cl2);
void show_means(int c);
bool check_mix_norm(int c);

// Evaluation
double evaluate();

// Global Variables 
Events::EventBigram bigram;   // Bigram LM
//Events::JointEventSet jes;
std::vector<int> sequence[S]; // array of sequences for all sentences
double PP;	 	      // Perplexity of the LM
int MixNumb;	              // current amount of mixtures we have
unsigned int seed;
bool initial_run;             // will indicate that it is the first run of the EM

std::vector<std::vector<std::vector<double> > > Means; 	     // means of the classes
std::vector<std::vector<double> >  MixWeights;               // mixture weights
std::vector<std::vector<std::vector<double> > > MixSignif;   // mixture significance
double Variances[D]; 	     				     // pooled variance
std::vector<std::vector<double> >  Q; 	     		     // backward recursion scores
std::vector<std::vector<double> >  Q_tilda;        	     // forward recursion scores
std::vector<std::vector<double> > P; 	     		     // prob(c/data) for the time step n
int number_ind[C]; 	     				     // indices for different numbers ( used for sampling a sequence)



int main(int argc, char* argv[]) 
{
	if(T < 0){
    		std::cerr<<" Negative amount of positions ! Stop! \n";
		return 0;
  	}


	/* 						Allocating of vectors 							*/
	
	// Allocate vectors for Means
  	Means.resize(C);
  	for (int i = 0; i < C; ++i) {
   		Means[i].resize(L);

    		for (int j = 0; j < L; ++j)
      			Means[i][j].resize(D);
  	}
	
	// Allocate vectors for MixWeights
  	MixWeights.resize(C);
  	for (int i = 0; i < C; ++i) 
   		MixWeights[i].resize(L);

	// Allocate vectors for Mixture Significance
  	MixSignif.resize(T);
  	for (int i = 0; i < T; ++i) {
   		MixSignif[i].resize(C);

    		for (int j = 0; j < C; ++j)
      			MixSignif[i][j].resize(L);
  	}
	

	// Allocate vector for the Q table
	Q.resize(T);
	for (int i = 0; i < T; ++i) 
   		Q[i].resize(C);

	// Allocate vector for the Q_tilda table
	Q_tilda.resize(T);
	for (int i = 0; i < T; ++i) 
   		Q_tilda[i].resize(C);

	// Allocate vector for the P table of probabilities
	P.resize(T);
	for (int i = 0; i < T; ++i) 
   		P[i].resize(C);

	
	
	// Timing
	std::clock_t start, end;
  	std::time_t tstart,tend;
	start = clock();
 	std::time(&tstart);


	/* 						Get the parameters from the command line 				*/

	unsigned int seed;        // seed for LM generation

	if(argc!=5) {
    	std::cout << std::endl << "EM for Unspupervised Training on a MNIST digits sampled from a bigram LM with a given perplexity" 
              << std::endl << std::endl
              << "Given the Perplexity the program generates bigram LM and the corresponding string" << std::endl
	      << " which will be the basis of EM for Unspupervised Training" << std::endl
              << std::endl << "Usage: " << argv[0]
              << " <stringFilename> <probDistFilename> <randomseed> <PP>"
              << std::endl << std::endl;
    	exit(EXIT_SUCCESS);
  	}

   	else 
		std::cout<<"\nThe program started for "<<S<<" sentences of "<<T<<" numbers.\n";

	std::string stringFileName;
  	std::string probDistFileName;

  	for(unsigned a=1; a<argc; a++) 
	{
    		std::istringstream is(argv[a]);
    		switch(a) {
    		case 1:
      			is >> stringFileName;
      			break;
    		case 2:
      			is >> probDistFileName;
     			 break;
    		case 3:
      			if(argv[a][0] == '-') {
        			seed = (unsigned) start;
      			}
      			else is >> seed;
      			break;
    		case 4:
     			is >> PP;
      			break;
    		}
  	}


	/*						 Initialise parameters ( means, variances, LM and MNIST pictures)         		*/
	
	std::cout<<"\nInitializing the parameters...\n";

	// initialize means and covariances for Gaussians	
	init_params(); 

	// Generate bigram LM with a given perplexity 
	double bigram_perplexity, string_perplexity;
	bigram_perplexity = generate_LM(seed);
	std::cout<<"Generated LM has perplexity : "<<bigram_perplexity<<"\n";

	int err_flag;


	/* 						 Write LM and sequence into a file 					*/
	
	std::ofstream stringFile;
  	std::ofstream probDistrFile;
	
	// open files for writting
  	stringFile.open(stringFileName.c_str(),std::ios::out);
	if(! stringFile.is_open())
  	{
		std::cerr<<"Cannot open the file for writting a sequence!\n";
		exit(EXIT_SUCCESS);
  	}
  	probDistrFile.open(probDistFileName.c_str(),std::ios::out);
	if(! probDistrFile.is_open())
  	{
		std::cerr<<"Cannot open the file for writting a bigram LM!\n";
		exit(EXIT_SUCCESS);
  	}

	// write the probability distribution into the file
  	bigram.printOn(probDistrFile); 

	// write all the sequences into the file
	stringFile << "# number of events: " << C << std::endl;
  	stringFile << "# amount of positions : " << T << std::endl;
	std::cout<<"Sampling "<<S<<" strings ...\n";
  	for(int sentence = 0; sentence < S; sentence++)
	{
		//std::cout<<"The sentence : "<<sentence+1<<" :\n";
			
		// Sample a sequence from the LM
		string_perplexity = sample_sequence(seed,sentence);
		// Write the sequence into a file
		stringFile << "\n\nString " << sentence << std::endl;
  		stringFile << "# Seed: " << seed << std::endl;
  		stringFile << "# perplexity (PP): " << string_perplexity << std::endl;
  		stringFile << "# sequence : \n";
  		for(int i=0; i<T;i++)
			stringFile << sequence[sentence][i]<<" ";
		seed++;
	}

	// close file for writting
  	stringFile.close();
  	probDistrFile.close();
	std::cout<<"Bigram probability distribution was written into file "<<stringFileName<<"\n";
	std::cout<<"All the sequences of events was written into file "<<probDistFileName<<"\n";


	/* 									TRAIN							*/
	EM();


	end = clock();
  	time(&tend);
  	std::cout << "#" << std::endl
              << "# TOTAL TIME:          " << difftime(tend,tstart) << " sec" 
              << " / from clock: "         << ((double) (end-start))/((double) CLOCKS_PER_SEC) << " sec" << std::endl
              << "#" << std::endl;

	return 0;
}

// It takes the folder with MNIST data as an argument
// and assumes that it has files data0, data1, data2 ...
bool read_mnist(std::vector<std::vector<std::vector<double> > >  MNIST, std::string folder)
{
	std::cout<<"Reading The MNIST dataset...\n";
	int n_rows=28;
        int n_cols=28;
	unsigned char temp=0;
	// Read all the numbers one by one
	for(int numb=0;numb<10;numb++)
	{
		std::ifstream curr_file((folder + "/data"+std::to_string(numb)).c_str());
    		if (curr_file.is_open())
    		{
        		for(int i=0;i<PictureAmount;++i)
        		{
            			for(int r=0;r<n_rows;++r)
           			{
                			for(int c=0;c<n_cols;++c)
                			{
                    				curr_file.read((char*)&temp,sizeof(temp));
						//std::cout<<"I am here\n";
		    				MNIST[numb][i][28*r + c] = int(temp);
                			}
            			}
        		}
		}
		else
			return true;
    		curr_file.close();
	}
	
	return false; // no errors
}


// Initialize parameters of all characters : arrays of means and variances
// It also generates LM
void init_params()
{
	// Initialize Gaussian parameters
  	srand (seed);
	std::default_random_engine generator;
  	std::uniform_real_distribution<double> distribution(0.0,Range);
	for(int d=0;d<D;d++)
		Variances[d] = Range;// / C;
	MixNumb=1; // we have only 1 mixture initially
	for(int c=0;c<C;c++)
		for(int d=0;d<D;d++)
			Means[c][0][d] = distribution(generator); 
	for(int c=0;c<C;c++)
		MixWeights[c][0] = 1;
	// Indicate that we are going to do the first run
	initial_run=true;
}

// Generate a bigram LM with a given Perplexity (PP) 
// return it's actual perplexity 
double generate_LM(unsigned seed)
{
	// Initialize bigram LM
	std::cout<<"Generating bigram LM ... \n";
	bigram = Events::EventBigram(C);
	double   bigram_perplexity, string_perplexity;

	// Simulate a bigram distribution with a givenPerplexity
  	int lm_restart = 0;
        do {
        	bigram.GetBigram(PP,seed); // Generate simulate probability distribution
		bigram_perplexity = bigram.calculatePerplexity() ; // calculate perplexity

        	lm_restart++;
		seed++;
        	if(lm_restart % 1000 == 0 && lm_restart > 1) std::cerr << "WARNING: LM restarted " << lm_restart << " times !" << std::endl;
        	if(lm_restart>100) seed++;


		//debug
		//std::cout<<"Bigram perplexity = "<<bigram_perplexity<<", string = "<<string_perplexity<<", needed perplexity = "<<PP<<"\n";

      	} while( fabs( bigram_perplexity -  PP) / PP > 0.005); //while(!bigram.FitStringPerplexity(PP,conditionalLogPP,T,seed) && std::abs(string_perplexity-PP)>epsilon);
	
	return bigram_perplexity;
}

// Sample a sequence using the bigram LM
// return it's perplexity 
double sample_sequence(unsigned seed, int sentence_ind)
{
	// Allocate the training sequence
  	sequence[sentence_ind].resize(T);
	//std::cout<<"Sampling a string with a perplexity : "<<PP<<" \n";
	double  string_perplexity;
	//std::vector<double> conditionalLogPP;
	// Create a joint event
	//jes = Events::JointEventSet(C);
	// Simulate a bigram distribution and a string with a givenPerplexity
  	int string_restart = 0;
	// Sample a string from the bigram LM untill the accuracy is enough
        do {

        	string_restart++;
		seed++;
        	if(string_restart % 1000 == 0 && string_restart > 1) std::cerr << "WARNING: LM restarted " << string_restart << " times !" << std::endl;
        	if(string_restart>100) seed++;

		// Generate a sequence from the probability distribution
		//string_perplexity = jes.AssignPrior(bigram,conditionalLogPP);
        	string_perplexity = bigram.generateString(sequence[sentence_ind],seed) ; //  and calculate string perplexity

		//debug
		//std::cout<<" String perplexity = "<<string_perplexity<<", needed perplexity = "<<PP<<"\n";

      	} while(fabs( string_perplexity -  PP) / PP > 0.5); //while(!bigram.FitStringPerplexity(PP,conditionalLogPP,T,seed) && std::abs(string_perplexity-PP)>epsilon);
	 // // || fabs( string_perplexity -  PP) / PP > 0.2);
	
	return string_perplexity;
}

void sample_MNIST(std::vector<std::vector<std::vector<double> > >  MNIST, int sentence_ind, std::vector<std::vector<double> > & x)
{
	//Sample accordingly to the given string
	int numb;
	for(int i=0; i < T; i++)
	{
		numb = sequence[sentence_ind][i];
		//std::cout<<"Numb : "<<numb;
		for(int d=0;d<D;d++)
	    		x[i][d] = MNIST[numb][number_ind[numb]][d];
    		number_ind[numb] = number_ind[numb] + 1;
		if(number_ind[numb] > PictureAmount ) // if we don't have any pictures more
		{
			std::cout<<"WARNING : Not enough pictures for the number "<<numb<<"! Started from the first one.\n";
			number_ind[numb] = 0; 
		}
	}
}


// HMM emission probability (logarithm of it) for a mixture "m" of class "cl"
double mix_emiss_prob(std::vector<double> x, int cl, int m)
{
	// use uniform distribution for the first run	
	if(initial_run)
		return 0; 
	double sum_for_exp = 0;
	double norm_fact = 0;
	// go over all dimensions
	for(int d =0; d<D;d++)
	{
		// Ignore not relevant features
		if(fabs(Variances[d]) < Range * 1.0 / 10000)
			continue;
		sum_for_exp += (x[d] - Means[cl][m][d]) * (x[d] - Means[cl][m][d]) / (Variances[d]*Variances[d]);
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

// Full emmision prob of a given class for a data x[D]
// In the log space !
double emiss_prob(std::vector<double> x, int cl)
{
	// Ignore mixture with too small weight
	int start_mix = 0;
	while(MixWeights[cl][start_mix] < epsilon) 
		start_mix++;
	//if(MixNumb == L)
	//	std::cout<<"MixWeights["<<cl<<"]["<<start_mix<<0] = "<<MixWeights[cl][0]<<", mix_emiss_prob(x,"<<cl<<",0) = "<<mix_emiss_prob(x,cl,0)<<".\n";
	double sum = log(MixWeights[cl][start_mix]) + mix_emiss_prob(x,cl,start_mix);
	if(MixNumb == 1) 
		return sum; // I want to Devide it by 100, because else it is way much more significant than transition prob.
	else
	{
		// Sum in the log space
		for (int l=start_mix + 1;l<MixNumb;l++)
			sum = add_log_scores(sum,mix_emiss_prob(x,cl,l) + log(MixWeights[cl][l]) );
		return sum;
	}
}


// Return a proportion of a mixture for a given time step and given class for given data
// gama(l | cl,Xd)
// but in a log-space
double mix_proportion(std::vector<double> x, int cl, int l)
{
	if(l > MixNumb)
		std::cerr<<"Error we tried to get a weight for mixture "<<l<<" which does not exist yet ( M = "<<MixNumb<<"\n";
	double total= emiss_prob(x, cl);
	double curr = log(MixWeights[cl][l])  + mix_emiss_prob(x,cl,l);
	
	// check normalization
	if(curr > total)
		std::cerr<<"One mixture has probability e^"<<curr<<" , while the the total one is e^"<<total<<" ! \nClass : "<<cl<<" mixture "<<l<<" weight : "<<MixWeights[cl][l]<<" \n";
	return curr - total;
}


// Transition probability from the generated LM
// In the log space
double MNIST_trans_prob(int prev, int next)
{
	double prob = bigram.GetProb(prev,next);
	if(prob > epsilon)
		return log(prob);
	else
		return - 5000;	
}

// Unigram probability from the generated LM
double MNIST_unigram_score(int number)
{
  	return MNIST_trans_prob(-1, number);	
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
int init_Q(std::vector<std::vector<double> > x)
{
/* input : training sequence */
	
	// At first time step we don't have predecessor, so we fall back to the unigram model
	for(int cl=0;cl<C;cl++) // go over all classe
	{
		Q[0][cl]  = emiss_prob(x[0],cl) + MNIST_unigram_score(cl);
		// check if the value make sence
		if(Q[0][cl] > epsilon)
		{
			std::cout<<"Error during initialization of a Q table : positive log-probability ( Q[0][cl] ) for class "<<cl<<"\n";
			std::cout<<"Emmision prob was "<<emiss_prob(x[0],cl)<<" transition was "<<MNIST_unigram_score(cl)<<"\n";
			return 0;
		}
		//std::cout<<"Emmision prob was "<<emiss_prob(x[0],cl)<<" transition was "<<MNIST_unigram_score(cl)<<"\n";
		//std::cout<<"During initialization emission prob. of class "<<cl<<" was : "<< emiss_prob(x[0],cl)<<"\n";
	}

	// All other values are initialized to -1 in order to see that it is not assigned the value yet
	for(int t=1;t<T;t++)
		for(int cl=0;cl<C;cl++) // go over all characters
			Q[t][cl]  = -1;
			
	//std::cout<<"For x[0] at t=0 : Emmision prob of class 3"<<emiss_prob(x[0],3)<<" transition was "<<MNIST_unigram_score(3)<<"\n";
	
	return 1;
}

// Forward algorithm
int forward(int t, int curr_char, std::vector<std::vector<double> > x)
{
// tries to calculate Q[n][curr_char] for our data 
	if(t==0) // we can use initialization
	{
		if(Q[0][curr_char] == -1)
			std::cerr<<" Error! Q was not initialized, but forward algorithm has started";
		return 0;
	}
	
	// in order to avoid numerical problems
	// we assign the sum to the first term
	double sum = Q[t-1][0] + MNIST_trans_prob(0, curr_char);  
	// sum over all other characters
	for(int prev_char=1;prev_char<C;prev_char++)
	{
		if(Q[t-1][prev_char] == -1)
			std::cerr<<" Error! At time step "<<t-1<<" for class "<<prev_char<<" Q was not initialized, but forward algorithm tried to use it\n";
		// We need to do summation in a logarithm space
		sum = add_log_scores(sum, Q[t-1][prev_char] + MNIST_trans_prob(prev_char, curr_char)); 
		// Check if we had numerical issues :
		if(sum == 10000)
		{
			std::cerr<<"Error occured while calculating forward part of Baum-Weill recursion for t = "<<t<<" curr_char = "<<curr_char<<" prev_char = "<<prev_char<<".\n";
			std::cerr<<"Q["<<t-1<<"]["<<prev_char<<"] = "<<Q[t-1][prev_char]<<" transition prob was "<< MNIST_trans_prob(prev_char, curr_char)<<"\n";
			return 0; // return error code
		}
	}
	Q[t][curr_char] = emiss_prob(x[t],curr_char) + sum;

	// check if the value make sence
	if(Q[t][curr_char] > 0)
	{
		std::cout<<"Error during forward path : positive log-probability at time step "<<t<<" for class "<<curr_char<<"\n";
		std::cout<<"Sum was "<<sum<<"\n";
		return 0;
	}
	
	// if we are here - we had no errors
	return 1;
}

// Initialization of a table Q_tilda
void init_Q_tilda(std::vector<std::vector<double> > x)
{
/* input : training sequence */
	
	for(int cl=0;cl<C;cl++) // go over all classe
		Q_tilda[T-1][cl]  = pow(10, -5000); // instead of 0 to avoid numerical problems

	// All other values are initialized to -1 in order to see that it is not assigned the value yet
	for(int n=0;n<T-1;n++)
		for(int cl=0;cl<C;cl++) // go over all characters
			Q_tilda[n][cl]  = -1;
	
}

// Backward algorithm
int backward(int n, int curr_char, std::vector<std::vector<double> > x)
{
	if(n==T-1) // we can use initialization
	{
		if(Q_tilda[T-1][curr_char] == -1)
			std::cerr<<" Error! Q_tilda was not initialized, but backward algorithm has started";
		return 0;
	}
	// in order to avoid numerical problems
	// we assign the sum to the first term
	// We know that the distribution at the last step is such, if next number is 1
	double sum = Q_tilda[n+1][0] + MNIST_trans_prob(curr_char, 0) + emiss_prob(x[n+1],0);  
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
		sum = add_log_scores(sum, Q_tilda[n+1][next_char] + MNIST_trans_prob(curr_char, next_char) + emiss_prob(x[n+1],next_char));  
		if( n == 2 && curr_char == 48)	
		{
			std::cout<<"For the next char "<<next_char<<" : Q prev = "<< Q_tilda[n+1][next_char]<<" trans prob ="<<MNIST_trans_prob(curr_char, next_char)<<" emiss = "<<emiss_prob(x[n+1],next_char);  
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
int Baum_algorithm(std::vector<std::vector<double> > x)
{
	int err_flag;

	// Initialise Q table
	err_flag = init_Q(x);

	// check numerical issues
	if(err_flag == 0)  
	{
		std::cerr<<"Have got numerical problem during initialization of the Q table!\n";
		return 0;
	}

	// Do forward path
	for(int n=1;n<T;n++){ // go over all timespeps
		for(int c=0;c<C;c++){ // go over all characters
			err_flag = forward(n,c,x);
			if(err_flag == 0)  // check numerical issues
				return 0;
		}
	}

	// Do backward path
	init_Q_tilda(x);
	for(int n=T-2;n>=0;n--){ // go over all timespeps
		for(int c=0;c<C;c++){ // go over all characters
			backward(n,c,x);
			if(err_flag == 0)  // check numerical issues
				return 0;
		}
	}
	// Calculate not normalized log - probabilities
	for(int n=0;n<T;n++) // go over all timespeps
		for(int c=0;c<C;c++) // go over all characters
			P[n][c] = (Q[n][c] + Q_tilda[n][c]);

	//debug

	/*for(int t=0;t<3;t++)
	{
		for(int c=0;c<C;c++)
		{
			std::cout<<"Q["<<t<<"]["<<c<<"] = "<<Q[t][c]<<"\n";
		}
		std::cout<<"\n";
	}



	
	for(int t=2;t<3;t++)
	{
		for(int c=0;c<C;c++)
		{
			std::cout<<"Q_tilda["<<t<<"]["<<c<<"] = "<<Q_tilda[t][c]<<"\n";
		}
		std::cout<<"\n";
	}	

	for(int t=2;t<3;t++)
	{
		for(int c=0;c<C;c++)
		{
			std::cout<<"P["<<t<<"]["<<c<<"] = "<<P[t][c]<<"\n\n";
		}
		std::cout<<"\n";
	}

	*/

	// Get a log of the total probabolity of a sequence for normalization
	double log_norm = word_total_probability();

	//std::cout<<"Normalization factor: "<<log_norm<<"\n";
	// Normalize the probabilities
	for(int n=0;n<T;n++) // go over all timespeps
		for(int c=0;c<C;c++) // go over all characters
			P[n][c] = P[n][c] -  log_norm;
	
	return 1;
}

/* 						Learning 						*/

// Expectation Maximization algorithm
void EM()
{
	double err_flag;
	std::vector<std::vector<std::vector<double> > >  MNIST;
	// Allocate vectors for MNIST pictures
  	MNIST.resize(C);
  	for (int i = 0; i < C; ++i) {
   		MNIST[i].resize(PictureAmount);

    		for (int j = 0; j < PictureAmount; ++j)
      			MNIST[i][j].resize(D);
  	}
	// read example observations 
	err_flag = read_mnist(MNIST,"MNIST_data");

	// check if the reading was succesfull
	if(err_flag)
	{
		std::cerr<<"Error with reading MNIST dataset!\n";
		return;
	}

	// Allocate vectors for sum_means
	std::vector<std::vector<std::vector<double> > > sum_means;
  	sum_means.resize(C);
  	for (int i = 0; i < C; ++i) {
   		sum_means[i].resize(L);

    		for (int j = 0; j < L; ++j)
      			sum_means[i][j].resize(D);
  	}
	
	// Allocate vectors for sum_gama_cl
	std::vector<std::vector<double> >  sum_gama_cl;
  	sum_gama_cl.resize(C);
  	for (int i = 0; i < C; ++i) 
   		sum_gama_cl[i].resize(L);

	// Allocate vector for sum_gama_c
	std::vector<double>  sum_gama_c;
	sum_gama_c.resize(C);

	// Allocate vector for sum_variances
	std::vector<double>  sum_variances;
	sum_variances.resize(D);

	//  Allocate vector for training data
	std::vector<std::vector<double> >  x; 
	x.resize(T);
  	for (int i = 0; i < T; ++i) 
   		x[i].resize(D);

	// Do EM iteration for the current amount of mixture densities
	for(int it=0;it<ITER;it++)
	{
		for(int i=0;i<10;i++)
			number_ind[i]=0;

		std::cout<<"I do "<<it+1<<" iteration of EM for "<<MixNumb<<" mixtures...\n";

		// initialize array of probabilities
		for(int c=0;c<C;c++)
		{
			for(int l=0;l<L;l++)
			{
				for(int d=0;d<D;d++)
					sum_means[c][l][d]=0;
				sum_gama_cl[c][l]=0;
			}
			sum_gama_c[c] = 0;
		}
		for(int d=0;d<D;d++)
			sum_variances[d] = 0;
		
		// Calculate the contribution to the sums from each sentence
		for(int sentence = 0; sentence < S; sentence++)
		{
			std::cout<<"Working with the sentence : "<<sentence+1<<"\n";

			sample_MNIST(MNIST,sentence, x); // Constuct training example			

			// Contribute to the sums
			addToSum(x, sum_means, sum_gama_cl, sum_gama_c, sum_variances);

			//debug
			/* for(int c=0;c<C;c++)
				std::cout<<"Sum for class "<<c<<" : "<<sum_gama_c[c]<<"\n"; */
			/*for(int c=0;c<C;c++)
				for(int l=0;c<;c++)
					 sum_gama_cl[c][l]*/
		}

		// Reestimate means
		for(int c=0;c<C;c++)
			for(int l=0;l<MixNumb;l++)
				for(int d=0;d<D;d++)
					if(sum_gama_cl[c][l] > epsilon) 
						Means[c][l][d] = sum_means[c][l][d] / sum_gama_cl[c][l];

		// Reestimate weights
		for(int c=0;c<C;c++) {
			if(sum_gama_c[c] < epsilon)
				continue;
			for(int l=0;l<MixNumb;l++) {
				MixWeights[c][l] =  sum_gama_cl[c][l] / sum_gama_c[c];
			}
		}

		// Check Mixture weighs for single densities
		if(MixNumb ==1)
			for(int c=0;c<C;c++)
				for(int l=0;l<MixNumb;l++)
					if(MixWeights[c][l] != 1) 
						std::cerr<<"Mixture weight at the beginning for class "<<c<<" mixture "<<l<<" is not one, but "<<MixWeights[c][l]<<" !\n";

		// Check Mixture weighs normalization
		for(int c=0;c<C;c++)
		{
			err_flag = check_mix_norm(c);
			if(err_flag == 0)
			{
				std::cout<<" Have got non-normalized Mixture Weights on "<<it<<" iteration fort the class "<<c<<"!\n";
				return;
			}
		}

		// Reestimate variances
		for(int d=0;d<D;d++)
			Variances[d] = sqrt(sum_variances[d]/S*T);


		// Regularization
		/*if(Variances[d] < 50)
			Variances[d] = 50;*/


		// Split
		if(it == ITER-1) // at the last iteration
		{
			if(MixNumb < L) // if we still have less means than we want:
			{
				std::cout<<"Splitting the means ...\n";
				for(int c=0;c<C;c++) // for all classes
				{
					for (int l=0;l<MixNumb;l++) // for all Gaussians in the mistures
					{
						for(int d=0;d<D;d++)
						{
							Means[c][l][d] = Means[c][l][d] * 0.98;
							Means[c][MixNumb + l][d] = Means[c][l][d] * 1.02;
						}
						MixWeights[c][l] = MixWeights[c][l]*1.0/2;
						MixWeights[c][MixNumb + l] = MixWeights[c][l]*1.0;
					}
				}
				MixNumb = MixNumb * 2;
				it = -1; // start the first iteration for the new means
			}
		}

		initial_run = false;
		
		// Evaluate
		double err_rate =  evaluate();
		std::cout<<"After "<<it+1<<" iteration for "<<MixNumb<<" mixtures : Error rate = "<<err_rate<<"%\n";
	}
}


// Add the contribution of the current sentence to the whole sum for gama_cl and gama_c
void addToSum(std::vector<std::vector<double> > x, std::vector<std::vector<std::vector<double> > > & sum_means, std::vector<std::vector<double> >  & sum_gama_cl, std::vector<double> & sum_gama_c, std::vector<double>  sum_variances)
{
	double sum;
	int err_flag;
	//std::cout<<"We learn "<<MixNumb<<" mixtures .\n";

	// first - get probabilities Pt(c / x1_N)
	err_flag = Baum_algorithm(x);
		
	// check if we had numerical problems
	if(err_flag == 0)
	{
		std::cout<<" Have got numerical problems during Baum-Welch algorithm!\n";
		return;
	}

	// check normalization
	err_flag = test_prob_norm();
	if(err_flag == 0)
	{
		std::cout<<" Have got non-normalized probability distributrion during EM algorithm!\n";
		return;
	}

	// Debug
	//show_means(6);
	//debug(data);

	// Add contribution of the sentence to the sums
	for(int c=0;c<C;c++) // for all classes
	{
		for (int l=0;l<MixNumb;l++) // for all Gaussians in the mistures
		{
			for(int t=0;t<T;t++) // go over all time-steps
			{
				MixSignif[t][c][l] = mix_proportion(x[t], c,l);
					
				//Check if mixture significance makes sence
				if(MixSignif[t][c][l] >0 ) 
					std::cerr<<"Mixture significance for time step "<<t<<" class "<<c<<" mixture "<<l<<"is more than one! It's = e^"<<MixSignif[t][c][l] <<"\n";
				if(std::isnan(MixSignif[t][c][l]))
					std::cerr<<"Mixture significance: "<<l<<" "<<" for class "<<c<<" at time step "<<t<<" :  is NAN ! \n";
	
				// Contribute to the sum_gama_cl
				sum_gama_cl[c][l]+=exp(P[t][c] + MixSignif[t][c][l]);

				// Contribute to the sum_means 
				for(int d=0;d<D;d++)
					for(int t=0;t<T;t++) 
						sum_means[c][l][d]+=exp(P[t][c] + MixSignif[t][c][l]) * x[t][d];

				// Contribute to the sum_variances
				for(int d=0;d<D;d++) 
					sum_variances[d]+=exp(P[t][c] + MixSignif[t][c][l] ) * (x[t][d] - Means[c][l][d])*(x[t][d] - Means[c][l][d]);
 			}
		}

		// Contribute to the gama_c
		for(int t=0;t<T;t++)
			sum_gama_c[c]+= exp(P[t][c]);
	}
}

			
	
// Log - probability of a given word
double word_total_probability()
{
	double sum=P[T-1][0]; // initialize the sum
	for(int c=1;c<C;c++)
		sum = add_log_scores(sum,P[T-1][c]); // sum in a log space
	return sum;
}

	


/* 					Testing and Debuging						*/

// Test if our probability distrubution for classes is normalized
int test_prob_norm()
{
	double sum, prob;
	//std::cout<<"I am testing the normalization of the probability distribution ...\n";
	for(int n=T-1;n<T;n++)
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
void debug(std::vector<std::vector<double> > x)
{
	/*
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
		*/
	
	int n=0;
	std::cout<<"\nTime step : "<<n<<" \n";
	for(int c=0;c<C;c++)
		std::cout<<"Q["<<n<<"]["<<c<<"] = "<<Q[n][c]<<"\n";
	for(int c=0;c<C;c++)
		std::cout<<"Q_tilda["<<n<<"]["<<c<<"] = "<<Q_tilda[n][c]<<"\n";
	for(int c=0;c<C;c++)
		std::cout<<"P["<<n<<"]["<<c<<"] = "<<P[n][c]<<"\n";

	/* std::cout<<"Sigma : ";
	for(int d=0;d<50;d++)
		std::cout<<Variances[d]<<" ";
	std::cout<<"\n";*/ 


	// DEBUG table Q
	// Find the highest prob at the first time-step
	/*
	double max_val=Q[0][0];
	int max=0;
	for(int c=0;c<C;c++)
	{
		// let's check Q now
		if(Q[T-1][c] > max_val)
		{
			max_val = Q[T-1][c];
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
	*/
	//std::cout<<"Q_tilda gives the most probable class for the time step "<<time_step<<" : "<<max<<" with the probability "<<max_val<<".\n";
}

void primitive_decode()
{
	double max_val; 
	int max;
	std::cout<<"Sequence of classes with the maximal score ";
	for(int n=0;n<T;n++) // for each time step
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
double evaluate()
{
	double err_rate=0;
	double max_val; 
	int max;
	int confusion[C][C];
	// initialize confusion matrix
	for(int c1=0;c1<C;c1++)
		for(int c2=0;c2<C;c2++)
			confusion[c1][c2] = 0;

	for(int sentence_ind=0; sentence_ind<S; sentence_ind++) // for each sentence
	{
		// sum up error rate
		for(int n=0;n<T;n++) // for each time step
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
			if(max != sequence[sentence_ind][n])
				err_rate++;
			confusion[sequence[sentence_ind][n]][max] ++; 
		}
	}

	// print confusion matrix
	std::cout<<"Confusion matrix : \n";
	for(int c1=0;c1<C;c1++)
	{
		for(int c2=0;c2<C;c2++)
			std::cout<<confusion[c1][c2]<<" ";
		std::cout<<"\n";
	}
	return err_rate * 100.0 / (T*S);
}

// Will calculate the L2-distance between means, normalised w.r.t. the range of values
double dist(int cl1, int cl2)
{
	double sum=0;
	for(int d=0;d<D;d++)
	{
		for(int l=0;l<MixNumb;l++)
		{
			sum+=(Means[cl1][l][d] - Means[cl2][l][d])*(Means[cl1][l][d] - Means[cl2][l][d]);
		}
	}
	sum = sqrt(sum) / (D * Range*MixNumb);
	return sum;
}

// Will show the means of the class "c"
void show_means(int c)
{
	std::cout<<"Rounded Means of class "<<c<<" : \n";
	for(int l=0;l<MixNumb;l++)
	{
		std::cout<<"Mixture  : "<<l<<"\n";
		for(int d2=0;d2<28;d2++) // second dimension of the image
		{
			for(int d1=0;d1<28;d1++)
			{
				if(Means[c][l][d1+d2*28] > Range/2)
					std::cout<<1<<" ";
				else std::cout<<0<<" ";
				//std::cout<<Means[3][d1+d2*16]<<" ";
			}
			std::cout<<"\n";
		}
	}
}

// Check if mixture weights sums up to one for the class "c"
bool check_mix_norm(int c)
{
	double sum=0;
	for(int l=0;l<MixNumb;l++)
	{
		sum+=MixWeights[c][l];
	}
	if(fabs(sum - 1) > 0.4)	
	{
		std::cout<<"The mixture weights sum for class "<<c<<" = "<<sum<<"\n";
		return 0;
	}
	else
		return 1;
}

		// Check if a few classes are two close to each other
		/*if(iter%2 == 0)
			continue; // do checking once in 2 time steps
		double diff; // difference between means
		for(int cl1=0;cl1<C;cl1++)
		{
			for(int cl2=cl1+1;cl2<C;cl2++)
			{
				diff = dist(cl1,cl2);
				if(cl1==0 && cl2 ==3)
					std::cout<<"Distance between 0 and 3 = "<<diff<<"\n";
				if(diff < 0.01) // if they are too close
				{
					std::cout<<"Classes "<<cl1<<" and "<<cl2<<" collapsed!\n";
					// Reset one of them to random values
	  				srand (time(NULL));
					std::default_random_engine generator;
  					std::uniform_real_distribution<double> distribution(0.0,1000);
					for(int d=0;d<D;d++)
						for(int l=0;l<MixNumb;l++)
							Means[cl1][l][d] = distribution(generator); // generate random number
					// Reset variances
					for(int d=0;d<D;d++)
						Variances[d] = 120;
				}
			}
		}*/
