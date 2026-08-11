//
// Created by Omkaar Sampigeadi on 8/8/2026.
//

#include <iostream>
#include <fstream>
#include <vector>


const int GRID_SIZE = 50;      // grid is GRID_SIZE x GRID_SIZE cells
const double D = 1.0;          // diffusion coefficient
const double DX = 1.0;         // spacing between grid cells
const double DT = 0.2;         // time step size
const int NUM_STEPS = 600;     // how many timesteps to simulate (longer, so gas has
                                // time to reach the doorway and leak into the 2nd room)
const int LOG_EVERY = 15;      // only writes snapshot every N steps (keeps CSV count manageable)

const int SOURCE_X = 25;       // where the source sits: centered in the LEFT room
const int SOURCE_Y = 12;       // (wall/doorway is at column 25)
const double SOURCE_RATE = 5.0; // how much concentration the source injects each step

using Grid = std::vector<std::vector<double>>;

// A WallGrid marks which cells are solid interior walls (true = wall,
// false = open air). Separate type from Grid so it's clear
// which one holds concentration values and which holds room geometry.
using WallGrid = std::vector<std::vector<bool>>;

Grid makeEmptyGrid () {
    return Grid(GRID_SIZE, std::vector<double>(GRID_SIZE, 0.0));
}

WallGrid makeTwoRoomLayout() {
    int wallCol = GRID_SIZE / 2;          // the dividing wall's column
    int doorStart = GRID_SIZE / 2 - 3;    // doorway is a small gap in the wall
    int doorEnd   = GRID_SIZE / 2 + 3;
    for (int i = 0; i < GRID_SIZE; i++) {
        bool isDoorway = (i >= doorStart && i <= doorEnd);
        if (!isDoorway) {
            walls[i][wallCol] = true;
        }
    }
    return walls;
}

// Performs ONE diffusion timestep: reads from `current`, writes the result
// into `next`. We need two separate grids because every cell's update
// depends on its neighbors' OLD values -- if we updated `current` in place,
// later cells in the loop would read already-updated neighbor values,
// which would corrupt the physics.

void diffuseStep(const Grid& current, const WallGrid& walls, Grid& next) {
    double alpha = D * DT / (DX * DX); // the D*dt/dx^2 factor from the formula

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            // Wall cells never hold gas -- keep them pinned at 0 and skip
            // the update entirely. They still matter to their open-air
            // neighbors via neighborValue() above.
            if (walls[i][j]) {
                next[i][j] = 0.0;
                continue;
            }

            double up    = neighborValue(current, walls, i, j, i - 1, j);
            double down  = neighborValue(current, walls, i, j, i + 1, j);
            double left  = neighborValue(current, walls, i, j, i, j - 1);
            double right = neighborValue(current, walls, i, j, i, j + 1);

            double laplacian = up + down + left + right - 4.0 * current[i][j];
            next[i][j] = current[i][j] + alpha * laplacian;
        }
    }
}