#ifndef EVENTS_HH
#define EVENTS_HH

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <climits>
#include <limits>
#include <vector>

namespace Events {
  typedef unsigned EventType;
  const unsigned Empty=0;

  
  // bigram language model

  class EventBigram {
  private:
    std::vector<double>               eventunigram;
    std::vector<double>               cumulativeunigram;
    std::vector<std::vector<double> > eventbigram;
    std::vector<std::vector<double> > cumulativebigram;
    bool                              bigram_available;

  public:
    // constructor for an empty bigram
    EventBigram();

    // constructor for a bigram with a fixed vocabulary of events and a given perplexity
    EventBigram(unsigned);
    
    unsigned size();

    // copy constructor
    EventBigram(const EventBigram& eb);
    
    // allocation
    const EventBigram& operator = (const EventBigram&);
    
    // assign language model bigram, incl. unigram (sentence start)
    double GetBigram(double,unsigned&);

    // draw event from uni/bigram - if predecessor argument is negative, use unigram, and bigram otherwise
    unsigned DrawEventGivenPredecessor(int, unsigned&);

    bool FitStringPerplexity(double, std::vector<double>&, unsigned, unsigned&);

    // compute unigram perplexity
    double GetUnigramPerplexity();

    // get bigram probability if pre is positive, and unigram prob. otherwise
    double GetProb(int pre,int event) ;

  private:
    // assign arbitrary distribution to unigram
    void SimulateUnigram(unsigned&);

    // assign bigram probabilities for given history and partial perplexity   
    double SimulateSingleMgram(std::vector<double>&,double,unsigned&);

    double SolvePerplexity(double, double, double, double, double);

  public:
    void printOn(std::ostream&) const;

    double calculatePerplexity() const;

    // Sample a string from the generated bigram / unigram distribution 
    double generateString(std::vector<int>& string, unsigned& seed) ;

    ~EventBigram();
  };

}

#endif



