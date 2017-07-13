CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra -g -Iinclude `pkg-config --cflags opencv` -I/usr/local/include/librealsense -I/usr/include/eigen3
LDFLAGS = `pkg-config --libs opencv` -lrealsense

EXE = main
SRCDIR = src
BINDIR = bin

OBJECTS = $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%.o,$(wildcard $(SRCDIR)/*.cpp))

all: $(EXE)

$(EXE): $(BINDIR) $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(EXE) $(LDFLAGS)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -MMD -o $@ $<

include $(wildcard $(BINDIR)/*.d)

$(BINDIR):
	mkdir -p $(BINDIR)

run:
	./$(EXE)

me_happy: all run

clean:
	rm -rf $(BINDIR) $(EXE)

.PHONY: clean all