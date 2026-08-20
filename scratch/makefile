RUN_ARGS := $(filter-out run, $(MAKECMDGOALS))

%:
	@:

compile-gcc:
	gcc -O3 -march=native -fopenmp -ffast-math -funroll-loops -falign-loops=32 -falign-functions=32 -ftree-vectorize -fvect-cost-model=very-cheap -g $(file) -o out/$(out)


compile-clang:
	@echo "NOT IMPLEMENTED YET"

compile-icx:
	@echo "NOT IMPLEMENTED YET"

run:
	out/$(bin) $(RUN_ARGS)
