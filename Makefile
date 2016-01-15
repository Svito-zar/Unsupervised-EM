SOURCE=OneToOne.cpp
MYPROGRAM=EM
MYINCLUDES=/work/smt2/nuhn/tools/build/kenlm-2015-07-28/include 
LINK=lkenlm 

MYLIBRARIES=/work/smt2/nuhn/tools/build/kenlm-2015-07-28/lib
CC=g++

#------------------------------------------------------------------------------



all: $(MYPROGRAM)



$(MYPROGRAM): $(SOURCE)

	$(CC) -I$(MYINCLUDES)-$(LINK) $(SOURCE) -o $(MYPROGRAM) -L$(MYLIBRARIES)

clean:

	rm -f $(MYPROGRAM)
