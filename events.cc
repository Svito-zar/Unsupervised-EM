// Provide joint distributions and corresponding operations

#include "events.hh"
#include "random.hh"
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_nan.h>

namespace Events {
  // --------------------------------------------------------------------
  // EventBigram definitions

  EventBigram::EventBigram() {
    bigram_available = false;
  }

  EventBigram::EventBigram(unsigned num_events) {
    unsigned i,j,size=num_events;
    double   puni;

    puni=1.0/((double) size);
    eventunigram.resize(size);
    cumulativeunigram.resize(size);
    eventbigram.resize(size);
    cumulativebigram.resize(size);
    for(i=0; i<size; i++) {
      eventunigram[i] = puni;
      cumulativeunigram[i] = puni + (i>0 ? cumulativeunigram[i-1] : 0);
      eventbigram[i].resize(size);
      cumulativebigram[i].resize(size);
      for(j=0; j<size; j++) {
        eventbigram[i][j] = puni;
        cumulativebigram[i][j] = puni + (i>0 ? cumulativebigram[i-1][j] : 0);
      }
    }
    bigram_available = true;
  }

  unsigned EventBigram::size() {
    return eventunigram.size();
  }

  // copy constructor
  EventBigram::EventBigram(const EventBigram& eb)
  {
    eventunigram      = eb.eventunigram;
    eventbigram       = eb.eventbigram;
    cumulativeunigram = eb.cumulativeunigram;
    cumulativebigram  = eb.cumulativebigram;
    bigram_available  = eb.bigram_available;
  }

  // allocation
  const EventBigram& EventBigram::operator = (const EventBigram& eb)
  {
    eventunigram      = eb.eventunigram;
    eventbigram       = eb.eventbigram;
    cumulativeunigram = eb.cumulativeunigram;
    cumulativebigram  = eb.cumulativebigram;
    bigram_available  = eb.bigram_available;

    return *this;
  }

  // assign language model bigram, incl. unigram (sentence start)
  double EventBigram::GetBigram(double perplexity,unsigned& seed) {
    Random::Ran2 rand(seed);
    unsigned N=eventunigram.size();
    double logPPsum,probabilitysum,PP;
    double logPP=-log(perplexity),logPPn,logN=log(N),low,high;
    double logPPuni,logPP_min,logPP_max;

    // Does not consider STRING perplexity!
    // SimulateUnigram(seed);

    // Considers STRING perplexity at least approximately:
    logPP_min = 2 * logPP;
    logPP_max = logPP_min + logN;

    // OPTIONAL: distribute perplexity uiniformly upon unigram and bigram:
    // logPP_min = logPP_max = logPP;
    // END of OPTION

    // std::cerr << "Before:   " << logPP_min << " (" << exp(-logPP_min) << ") / "
    //           << logPP_max << " (" << exp(-logPP_max) << ")" << std::endl;
    if(logPP_min < -logN) logPP_min = -logN;
    if(logPP_max > 0)     logPP_max = 0;
    // std::cerr << "After:    " << logPP_min << " (" << exp(-logPP_min) << ") / "
    //           << logPP_max << " (" << exp(-logPP_max) << ")" << std::endl;

    logPP = logPP_min+rand.work()*(logPP_max-logPP_min);
    logPPuni = -2*log(perplexity) - logPP;
    // std::cerr << "Uni:      " << logPPuni << " (" << exp(-logPPuni) << ")" << std::endl;
    logPPuni = SimulateSingleMgram(eventunigram,logPPuni,seed);
    // std::cerr << "          " << logPPuni << " (" << exp(-logPPuni) << ")" << std::endl;
    logPPsum=0;
    probabilitysum=0;
    for(unsigned n=0; n<N; n++) {
      probabilitysum += eventunigram[n];
      if(eventunigram[n]>0) {
        low = (logPP - logPPsum)/eventunigram[n];
        if(low < -logN) low = -logN;
        high = (logPP + (1-probabilitysum) * logN - logPPsum)/eventunigram[n];
        if(high > 0) high = 0;
      }
      else {
        low = high = -log(perplexity);
      }
      
      // double ddd;
      logPPn = SimulateSingleMgram(eventbigram[n],low+rand.work()*(high-low),seed);
      // std::cerr << ddd << " --- " << logPPn << std:: endl;
      logPPsum += eventunigram[n] * logPPn;
    }
        
    bigram_available = true;

    cumulativeunigram[0] = eventunigram[0];
    for(unsigned i=1; i<N; i++) {
      cumulativeunigram[i] = cumulativeunigram[i-1] + eventunigram[i];
    }
    for(unsigned i=0; i<N; i++) {
      cumulativebigram[i][0] = eventbigram[i][0];
      for(unsigned j=1; j<N; j++) {
        cumulativebigram[i][j] = cumulativebigram[i][j-1] + eventbigram[i][j];
      }
    }

    // BEWARE: seed should be drawn before return!
    seed = (unsigned) (rand.work() * ((double) std::numeric_limits<unsigned>::max()));
    
    return exp(-(logPPsum + logPPuni)/2);
  }

  // draw event from uni/bigram - if predecessor argument is negative, use unigram, and bigram otherwise
  unsigned EventBigram::DrawEventGivenPredecessor(int predecessor, unsigned& seed) {
    Random::Ran2 rand(seed);
    double rnd=rand.work();
    int i;

    // std::cerr << "rnd = " << rnd << "  pre=" << predecessor << ": ";
    if(predecessor < 0) {
      for(i=0; i<eventunigram.size() && cumulativeunigram[i]<rnd; i++);
    }
    else {
      for(i=0; i<eventunigram.size() && cumulativebigram[predecessor][i]<rnd; i++);
    }
    if(i>=eventunigram.size()) {
      i=eventunigram.size()-1;
      std::cerr << "WARNING: end of cumulative uni-/bigram reached!" << std::endl;
    }

    // std::cerr << " drawn event = " << i << std::endl;

    seed = (unsigned) (rand.work() * ((double) std::numeric_limits<unsigned>::max()));

    return i;
  }

  bool EventBigram::FitStringPerplexity(double perplexity, std::vector<double>& conditionalLogPP, unsigned num_positions, unsigned& seed) {
    unsigned num_events(eventunigram.size()),event_Emax;
    Random::Ran2 rand(seed);
    double Econdmin(exp(-conditionalLogPP[0])),Emin(conditionalLogPP[0]),Emax(conditionalLogPP[0]),Eaverage(conditionalLogPP[0]),checkPP;
    double logPP(-((double) num_positions)*log(perplexity)),norm(0.0);
    std::vector<double> uni_min,uni_max;
    double E;

    uni_min.resize(num_events,0);
    uni_max.resize(num_events,0);
    uni_min[0] = exp(-conditionalLogPP[0]);

    // OPTIONAL: initialize with random unigram    
    // while(norm==0.0) {
       // norm=0.0;
       // for(unsigned n=1; n<num_events; ++n) {
         // eventunigram[n]=rand.work();
         // norm+=eventunigram[n];
         // }    
       // }
    // for(unsigned n=1; n<num_events; ++n) eventunigram[n]/=norm;
    // END of OPTION

    checkPP = (eventunigram[0]>0 ? eventunigram[0] * log(eventunigram[0]) : 0) + eventunigram[0] * conditionalLogPP[0];
    event_Emax = 0;
    for(unsigned n=1; n<num_events; ++n) {
      if(Emin>conditionalLogPP[n]) Emin=conditionalLogPP[n];
      if(Emax<conditionalLogPP[n]) {
        Emax=conditionalLogPP[n];
        event_Emax = n;
      }
      Eaverage+=conditionalLogPP[n];
      Econdmin += exp(-conditionalLogPP[n]);
      uni_min[n] = exp(-conditionalLogPP[n]);
      checkPP += (eventunigram[n]>0 ? eventunigram[n] * log(eventunigram[n]) : 0) + eventunigram[n] * conditionalLogPP[n];
    }
    Emin-=log(num_events);
    Eaverage/=((double) num_events);
    for(unsigned n=0; n<num_events; ++n) uni_min[n]/=Econdmin;
    Econdmin=-log(Econdmin);
    // std::cerr << "FitStringPerplexity:" << std::endl;
    // std::cerr << "PPcheck      = " << exp(-checkPP/num_positions)  << std::endl;
    // std::cerr << "PP min       = " << exp(-Econdmin/num_positions) << std::endl;
    // std::cerr << "PP Emin      = " << exp(-Emin/num_positions)     << std::endl;
    // std::cerr << "PP Emax      = " << exp(-Emax/num_positions)     << std::endl;
    // std::cerr << "PP Eaverage  = " << exp(-Eaverage/num_positions) << std::endl << std::endl;

    uni_max[event_Emax] = 1;
    if(checkPP>logPP) {
      uni_max=eventunigram;
    }
    else {
      uni_min=eventunigram;
    }
    
    // NEXT: Check, if desired PP is in the interval
    // if no:  return false -> loops bigram generation
    // if yes: get solution and return true

    if(Econdmin<=logPP && logPP<=Emax) {
      // get solution by greedy iteration
      double difference(1);
      while(difference>1e-8) {
        // double epsilon = rand.work();
        double epsilon(0.5);
        E = 0;
        for(unsigned n=0; n<num_events; ++n) {
          eventunigram[n] = uni_min[n] + epsilon * (uni_max[n] - uni_min[n]);
          E += (eventunigram[n]>0 ? eventunigram[n] * log(eventunigram[n]) : 0) + eventunigram[n] * conditionalLogPP[n];
        }
        if(E>logPP) {
          uni_max = eventunigram;
        }
        else{
          uni_min = eventunigram;
        }
        difference = 0;
        for(unsigned n=0; n<num_events; ++n) difference += std::abs(uni_max[n]-uni_min[n]);
      }
      seed = (unsigned) (rand.work() * ((double) std::numeric_limits<unsigned>::max()));
      if(std::abs(E-logPP)>1e-5) {
        std::cerr << std::endl << "WARNING: target perplexity" << perplexity << " not reached: "
                  << exp(-E/num_positions) << std::endl;
      }

      cumulativeunigram[0] = eventunigram[0];
      for(unsigned i=1; i<num_events; i++) {
        cumulativeunigram[i] = cumulativeunigram[i-1] + eventunigram[i];
      }

      return true;
    }
    else {
      seed = (unsigned) (rand.work() * ((double) std::numeric_limits<unsigned>::max()));
      return false;
    }
  }

  double EventBigram::GetUnigramPerplexity()
  {
    double logPP(0.0);
    unsigned n;
    
    for(unsigned n=0; n<eventunigram.size(); ++n) logPP -= (eventunigram[n]>0 ? eventunigram[n] * log(eventunigram[n]) : 0);
      
    return exp(logPP);
  } 
  
  double EventBigram::GetProb(int pre,int event) 
  {
    if(pre<0) return eventunigram[event];
    else {
      if(event<0) return 1;
      else return eventbigram[pre][event];
    }
  }

  
  void EventBigram::SimulateUnigram(unsigned& seed) {
    Random::Ran2 rand(seed);
    unsigned N=eventunigram.size(),ir;
    double balance=1,d;

    for(unsigned n=1; n<N; n++) {
      d = balance * rand.work();
      balance -= d;
      eventunigram[n] = d;
    }
    eventunigram[0] = balance; // normalization: remainder

    // regroup probabilities stochastically
    for(unsigned n=N-1; n>0; --n) {
      ir=((unsigned) (((double) n+1) * rand.work()));
      if(ir<n) {
	double d=eventunigram[n];
        eventunigram[n]  = eventunigram[ir];
        eventunigram[ir] = d;
      }
    }    
  }


  // assign bigram probabilities for given history and partial perplexity
  double EventBigram::SimulateSingleMgram(std::vector<double>& mgram, double logPP,unsigned& seed) {
    Random::Ran2 rand(seed);
    unsigned N=mgram.size(),ir;
    double probabilitysum=0, logPPsum=0, puni=1/((double) N);
    double a,b,c,x1,x2,x3,x4,f1max,f2max,flowermin,fuppermin;
    double low1,low2,high1,high2,interval_length,pn;
    double epsilon=1e-6;

    if(logPP<=-log((double) N)*(1-epsilon)) { // NOTE: otherwise, maximum PP=N numerically can lead to errors solving perplexity problems below!!
      for(unsigned n=0; n<N; ++n) {
        mgram[n] = puni;
      }
      logPPsum = log(puni);
    }
    else {
      for(unsigned n=1; n<N; n++) {
        a = 1.0 - probabilitysum;
        b = 1.0/(N-n);
        c = logPP - logPPsum;
        f2max = (a>0) ? a*log(a) : 0; 
        f1max = f2max + a*log(b);
        flowermin = f1max - a*log(1+b);
        fuppermin = f2max + a*log(0.5);
        
        if(f1max<=c) x1=0;
        else if(flowermin>=c) x1 = a*b / (1+b);
        // else if(flowermin>c) x1=a+1;
        else x1 = SolvePerplexity(a,b,c,0,a*b/(1+b));
        
        if(flowermin>=c) x2=x1;
        // else if(flowermin>c) x2=a+1;
        else if(f2max<=c) x2=a; 
        // Note: in case of f2max<c this would be an error,
        //       nontheless, the error is caught below for x3,x4
        else x2 = SolvePerplexity(a,b,c,a*b/(1+b),a);
        
        if(f2max<c) x3=a+1;
        else if(f2max==c) x3=0;
        else if(fuppermin>=c) x3=-1;
        else x3 = SolvePerplexity(a,1.0,c,0,0.5*a);
        
        if(fuppermin>=c) x4=a+1;
        else if(f2max<c) x4=-1;
        else if(f2max==c) x4=a;
        else x4 = SolvePerplexity(a,1.0,c,0.5*a,a);
        
        // std::cerr << n << " : " << x1 << " " << x2 << " " << x3 << " " << x4 << " " << f1max << " " << f2max << " " << flowermin << " " << fuppermin << " " << c << " ";
        if(x1>x3 || x2<x4) {
          // cases fuppermin > c, or
          //       x1 > x3, or
          //       x2 < x4
          low1 = std::max(x1,x3);
          high1 = std::min(x2,x4);
          interval_length = high1 - low1;
          low2 = high2 = high1 + 1;
          // std::cerr << "Case 1: " << low1 << " " << high1 << " ";
        }
        else {
          // two intervals to be drawn from:
          // [x1,x2] intersected with [0,x3], and
          low1 = std::max(x1,(double) 0);
          high1 = std::min(x2,x3);
          // [x1,x2] intersected with [x4,a]
          low2 = std::max(x1,x4);
          high2 = std::min(x2,a);
          interval_length = high1 - low1 + high2 - low2;        
          // std::cerr << "Case 2: " << low1 << " " << high1 << " ";
          // std::cerr << low2 << " " << high2 << " ";
        }
        // catch cases low1=high1 and/or low2=high2??
        // double ddd;
        pn = low1 + interval_length * (rand.work());
        if(pn > high1) pn += low2 - high1;
        // std::cerr << pn << " rand = " << ddd << std::endl;
        
        probabilitysum += pn;
        logPPsum += (pn>0) ? pn*log(pn) : 0;
        mgram[n-1] = pn;      
      }
      pn = 1 - probabilitysum;
      logPPsum += (pn>0) ? pn*log(pn) : 0;
      mgram[N-1] = pn;  

      // regroup probabilities stochastically
      for(unsigned n=N-1; n>0; --n) {
        ir=((unsigned) (((double) n+1) * rand.work()));
        if(ir<n) {
          double d=mgram[n];
          mgram[n]  = mgram[ir];
          mgram[ir] = d;
        }
      }
    }
    

    seed = (unsigned) (rand.work() * ((double) std::numeric_limits<unsigned>::max()));

    bigram_available = false;
    return logPPsum;
  }

  double EventBigram::SolvePerplexity(double a, double b, double c, double low, double high) {
    // call only if checked that solution is possible, 
    // i.e. if a*log(a) >= c >= (a*log ab/(1+b))
    unsigned i = 10000;
    // length of maximum co-domain serves as basis for optimization threshhold:
    double t = std::abs(1e-16*c);
    double x;
    double xmin = low;
    double xmax = high;
    double f,fmin,fmax;

    fmin  = ((xmin>0) ? xmin*log(xmin) : 0) + ((xmin<a) ? (a-xmin)*log(b*(a-xmin)) : 0) - c;
    fmax  = ((xmax>0) ? xmax*log(xmax) : 0) + ((xmax<a) ? (a-xmax)*log(b*(a-xmax)) : 0) - c;

    if(fmin == 0) {
      fmax = fmin;
      xmax = xmin;
    }
    else if(fmax == 0) {
      fmin = fmax;
      xmin = xmax;
    }
    else if(fmin * fmax > 0) {
      std::cerr << "ERROR in EventBigram::SolvePerplexity:" << std::endl;
      std::cerr << "no sign change within interval [";
      std::cerr << low << "," << high << "]" << std::endl;
      std::cerr << "with a = " << a << ", b = " << b << ", c = " << c << std::endl;
      exit(EXIT_FAILURE);
    }

    x = xmin;
    f = fmin;    
    while(i-- > 0 && (std::abs(f) > t) && xmin != xmax) {
      if( (f > 0 && fmin > 0) || (f < 0 && fmin < 0) ) {
        xmin = x;
        fmin = f;
      }
      else {
        xmax = x;
        fmax = f;
      }

      x = 0.5 * (xmin + xmax);
      f  = ((x>0) ? x*log(x) : 0) + ((x<a) ? (a-x)*log(b*(a-x)) : 0) - c;      
    } 

    // if(std::abs(f)>t) {
    //   std::cerr << "WARNING: bracketing threshhold not met:" << std::endl;
    //   std::cerr << "     t  = " << t << std::endl;
    //   std::cerr << "  < |f| = " << std::abs(f) << std::endl;
    // }

    return x;
  }

  void EventBigram::printOn(std::ostream& strm) const {
    unsigned i,j;
    double sum,logPPsum=0,logPPpartialsum;
    
    strm << eventunigram.size() << " :" << std::endl;
    for(sum=0,i=0; i<eventunigram.size(); ++i) {
      strm << " p(" << i << ") = " << eventunigram[i] << " cum: " << cumulativeunigram[i] << std::endl;
      sum += eventunigram[i];
    }
    strm << " sum_j p(j) = " << sum << std::endl << std::endl;
    for(i=0; i<eventunigram.size(); ++i) {
      for(logPPpartialsum=0,sum=0,j=0; j<eventunigram.size(); ++j) {
        strm << " p(" << j << "|" << i << ") = " << eventbigram[i][j] << " cum: " << cumulativebigram[i][j] << std::endl;
        sum += eventbigram[i][j];
        logPPpartialsum += (eventbigram[i][j]>0) ? eventbigram[i][j]*log(eventbigram[i][j]) : 0;
      }
      strm << " sum_j p(j|" << i << ") = " << sum << " ";
      strm << " partial PP = " << exp(-logPPpartialsum) << std::endl;
      logPPsum += eventunigram[i] * logPPpartialsum;
    }    
    strm << " total PP = " << exp(-logPPsum) << std::endl;
  }

  double EventBigram::calculatePerplexity() const {
    unsigned i,j;
    double sum,logPPsum=0,logPPpartialsum;

    for(sum=0,i=0; i<eventunigram.size(); ++i) {
      sum += eventunigram[i];
    }

    for(i=0; i<eventunigram.size(); ++i) {
      for(logPPpartialsum=0,sum=0,j=0; j<eventunigram.size(); ++j) {
        sum += eventbigram[i][j];
        logPPpartialsum += (eventbigram[i][j]>0) ? eventbigram[i][j]*log(eventbigram[i][j]) : 0;
      }
      logPPsum += eventunigram[i] * logPPpartialsum;
    }    

     return exp(-logPPsum);
  }

  // Sample a string from the generated bigram / unigram distribution 
  // Return perplexity of this string
  double EventBigram::generateString(std::vector<int>& string, unsigned& seed) {
	unsigned prev,curr;
	int length = string.size();
	double logPP, stringProbLog;
	if(length == 0) 
	{
		std::cerr<<" EventBigram::generateString : Input string must be not empty! \n";
		return 1;
	}

	// Sample a first event of a string from the unigram distribution
	prev = -1;
        curr = DrawEventGivenPredecessor(prev,seed);
	string[0] = curr;
	stringProbLog = log(eventunigram[curr]); // we need to calculate in the log space because of the numerical issues

	// Sample the whole string from the bigram distribution
	prev = curr;
	for(int i=1;i<length;i++)
	{
		//std::cout<<"String log-probability : "<<stringProbLog<<"\n";
		curr = DrawEventGivenPredecessor(prev,seed);
		string[i] = curr;
		stringProbLog = stringProbLog + log(eventbigram[prev][curr]) ;
		prev = curr;
	}

	//std::cout<<"String log-probability : "<<stringProbLog<<"\n";
	logPP = -stringProbLog/length;
	return exp(logPP); // PP
  }

  EventBigram::~EventBigram() {
    eventunigram.clear();
    eventbigram.clear();
  }
}

