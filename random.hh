#ifndef _RANDOM_HH_
#define _RANDOM_HH_

#include <vector>
#include <cmath>

namespace Random { 

  /** Random number generator
   *  (Copied from Numerical Recepies in C++)
   *  Properties:
   *   -Simple and fast
   *   -Fails on some statistical tests
   *   -Period: 2^31 - 2 ~ 2.1e9
   */
  class Ran0 {
  private:
    static const int IM;
  private:
    int idum_;
  private:
    void next() {
      const int IA=16807,IQ=127773,IR=2836;
      int k=idum_/IQ;
      idum_=IA*(idum_-k*IQ)-IR*k;
      if (idum_ < 0) idum_ += IM;
    }
  public:
    Ran0(int idum);
    double work();
  };
  
  /** Random number generator
   *  (Copied from Numerical Recepies in C++)
   *  Properties:
   *   -Passes those statistical tests that Ran0 is known to fail if number of calls < 10^8
   */
  class Ran1 {
  private:
    static const int NTAB;
    static const int IM;
  private:
    int idum_;
    int iy_;
    std::vector<int> iv_;
  private:
    void next() {
      const int IA=16807,IQ=127773,IR=2836;
      int k=idum_/IQ;
      idum_=IA*(idum_-k*IQ)-IR*k;
      if (idum_ < 0) idum_ += IM;
    }
  public:
    /** Initializes the random generator by @param idum
     *  @param idum is positive integer, unlike in "Numerical Recepies in C++".
     */
    Ran1(int idum);
    double work();
  };
  
  /** Random number generator
   *  (Copied from Numerical Recepies in C++)
   *  Properties:
   *   -"Perfect" random number generator for floating point precision
   *   -Period: ~ 2e18
   */
  class Ran2 {
  private:
    static const int NTAB;
    static const int IM1;
  private:
    int idum_;
    int idum2_;
    int iy_;
    std::vector<int> iv_;
  private:
    void next() {
      const int IA1=40014,IQ1=53668,IR1=12211;
      int k=idum_/IQ1;
      idum_=IA1*(idum_-k*IQ1)-k*IR1;
      if (idum_ < 0) idum_ += IM1;
    }
    void next2() {
      const int IA2=40692,IQ2=52774,IR2=3791;
      const int IM2=2147483399;
      
      int k=idum2_/IQ2;
      idum2_=IA2*(idum2_-k*IQ2)-k*IR2;
      if (idum2_ < 0) idum2_ += IM2;
    }
  public:
    /** Initializes the random generator by @param idum
     *  @param idum is positive integer, unlike in "Numerical Recepies in C++".
     */
    Ran2(int idum);
    double work();
  };
  
  /** Random number generator
   *  (Copied from Numerical Recepies in C++)
   *  Knuth suggestion, see also Knuth: The Art of Computer Programming, Vol 2.
   */
  class Ran3 {
  private:
    static const int MBIG;
    static const int MZ;
  private:
    int idum_;
    int inext_;
    int inextp_;
    std::vector<int> ma_;
  public:
    /** Initializes the random generator by @param idum
     *  @param idum is positive integer, unlike in "Numerical Recepies in C++".
     */
    Ran3(int idum);
    double work();
  };
  
  /** Random number generator with normal distribution
   *  Template parameter Ran is type of the uniform random generator.
   */
  template<class Ran>
  class Gasdev {
    Ran ran_;
    int iset_;
    double gset_;
  public:
    /** Initializes the random generator by @param idum
     *  @param idum is positive integer, unlike in "Numerical Recepies in C++".
     */
    Gasdev(int idum);
    double work();
  };
  
  template<class Ran>
  Gasdev<Ran>::Gasdev(int idum) :
    ran_(idum), iset_(0), gset_(0)
  {}
  
  template<class Ran>
  double Gasdev<Ran>::work()
  {
    double fac,rsq,v1,v2;

    if (iset_ == 0) {
      do {
	v1=2.0*ran_.work()-1.0;
	v2=2.0*ran_.work()-1.0;
	rsq=v1*v1+v2*v2;
      } while (rsq >= 1.0 || rsq == 0.0);
      fac=sqrt(-2.0*log(rsq)/rsq);
      gset_=v1*fac;
      iset_=1;
      return v2*fac;
    } else {
      iset_=0;
      return gset_;
    }
  }
} 

#endif // _RANDOM_HH_
