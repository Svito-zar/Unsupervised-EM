OBJS	=	events.o	\
		random.o

LIBS	=	-lgsl -lgslcblas -lm 

CC=g++
CFLAGS=-std=c++0x -O0 -g

default	:	${OBJS}	EM.o
	$(CC) $(CFLAGS) -o  EM ${OBJS} EM.o $(LIBS)

.SUFFIXES	:	.o .cc

.cc.o	:
	$(CC) $(CFLAGS) -c $<

clean	:
	rm *.o EM

# DO NOT DELETE

EM.o: events.hh
EM.o: events.hh
events.o: events.hh random.hh
random.o: random.hh 





#$(CC) $(CFLAGS)

#SOURCE=OneToOne.cc




#------------------------------------------------------------------------------



#all: $(MYPROGRAM)



#$(MYPROGRAM): $(SOURCE)

#	$(CC) $(CFLAGS) $(SOURCE) -o $(MYPROGRAM) $(LIBS)

