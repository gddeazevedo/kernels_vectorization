BUILD_DIR := build
OPERATION := $(word 2,$(MAKECMDGOALS))

GCC_FLAGS := -O3 -march=native -fopenmp -ffast-math -funroll-loops -falign-loops=32 -falign-functions=64
ICX_FLAGS := -O3 -march=native -qopenmp -ffast-math -funroll-loops -falign-loops=32 -falign-functions=64

up:
	docker compose up -d

down:
	docker compose down

bash:
	docker compose exec kernels_vectorization bash

gcc: clean
	cmake -B $(BUILD_DIR) -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS="$(GCC_FLAGS)"
	cmake --build $(BUILD_DIR)
	./$(BUILD_DIR)/kernels_vectorization $(OPERATION) gcc

icx: clean
	cmake -B $(BUILD_DIR) -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS='$(ICX_FLAGS)'
	cmake --build $(BUILD_DIR)
	./$(BUILD_DIR)/kernels_vectorization $(OPERATION) icx

clean:
	rm -rf $(BUILD_DIR)

%:
	@:

.PHONY: gcc icx clean up down bash