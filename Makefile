SOURCE=OneToOne.cpp
MYPROGRAM=EM
MYINCLUDES=/work/smt2/nuhn/tools/build/kenlm-2015-07-28/include 
LINK=lkenlm 

MYLIBRARIES=/work/smt2/nuhn/tools/build/kenlm-2015-07-28/lib
CC=g++
CFLAGS=-std=c++11 

#------------------------------------------------------------------------------



all: $(MYPROGRAM)



$(MYPROGRAM): $(SOURCE)

	$(CC) $(CFLAGS) -I$(MYINCLUDES)-$(LINK) $(SOURCE) -o $(MYPROGRAM) -L$(MYLIBRARIES)

clean:

	rm -f $(MYPROGRAM)
