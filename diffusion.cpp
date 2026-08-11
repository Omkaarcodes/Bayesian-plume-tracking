//
// Created by Omkaar Sampigeadi on 8/8/2026.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>   // for creating the per-run folder
#include <chrono>       // for getting the current time
#include <ctime>        // for formatting the current time as a string
#include <sstream>
#include <iomanip>      // for std::put_time


const int GRID_SIZE = 50;      // grid is GRID_SIZE x GRID_SIZE cells
const double D = 1.0;          // diffusion coefficient
const double DX = 1.0;         // spacing between grid cells
const double DT = 0.2;         // time step size
const int NUM_STEPS = 600;     // how many timesteps to simulate (longer, so gas has
                                // time to reach the doorway and leak into the 2nd room)
const int LOG_EVERY = 15;      // only writes snapshot every N steps (keeps CSV count manageable)

const int SOURCE_X = 35;       // where the source sits: in the kitchen
const int SOURCE_Y = 38;       // (a stove-leak-style source, away from the doorway)
const double SOURCE_RATE = 5.0; // how much concentration the source injects each step

using Grid = std::vector<std::vector<double>>;

// A WallGrid marks which cells are solid interior walls (true = wall,
// false = open air). Separate type from Grid so it's clear
// which one holds concentration values and which holds room geometry.
using WallGrid = std::vector<std::vector<bool>>;

// A fixed sensor sitting at one grid cell. It doesn't need to remember its
// own reading history as a field.
struct Sensor {
    int id;
    int x;
    int y;
};

// Places one sensor in each room of the house, roughly centered, plus one
// in the hallway.
std::vector<Sensor> placeSensors() {
    return {
            {0, 12, 11},   // bedroom
            {1, 37, 11},   // living room
            {2, 7, 37},    // bathroom
            {3, 44, 44},   // kitchen, far corner from the leak source
            {4, 25, 24}    // hallway
    };
}

Grid makeEmptyGrid() {
    return Grid(GRID_SIZE, std::vector<double>(GRID_SIZE, 0.0));
}

WallGrid makeEmptyWalls() {
    return WallGrid(GRID_SIZE, std::vector<bool>(GRID_SIZE, false));
}

// Fills every cell in the specified range as wall.
// Used to lay down straight wall segments -- outer walls, dividers, etc.
void addWallSegment(WallGrid& walls, int iStart, int iEnd, int jStart, int jEnd) {
    for (int i = iStart; i <= iEnd; i++) {
        for (int j = jStart; j <= jEnd; j++) {
            walls[i][j] = true;
        }
    }
}

// Clears a rectangular gap in a wall so two spaces connect. Same shape as
// addWallSegment but sets false instead of true -- meant to be called AFTER
// a wall segment, to carve a doorway out of it.
void addDoorway(WallGrid& walls, int iStart, int iEnd, int jStart, int jEnd) {
    for (int i = iStart; i <= iEnd; i++) {
        for (int j = jStart; j <= jEnd; j++) {
            walls[i][j] = false;
        }
    }
}

// Builds a small house: a central hallway (columns 23-26) running north-south,
// with four rooms opening off it -- bedroom and living room to the west,
// bathroom and kitchen to the east -- plus a front door on the south wall
// that opens directly onto the hallway. Every room connects to the world
// only through its own doorway onto the hallway, the way an actual hallway-
// plan house works, rather than rooms opening into each other directly.
WallGrid makeHouseLayout() {
    WallGrid walls = makeEmptyWalls();

    // outer walls -- the house's exterior envelope
    addWallSegment(walls, 0, 0, 0, GRID_SIZE - 1);                         // north wall
    addWallSegment(walls, GRID_SIZE - 1, GRID_SIZE - 1, 0, GRID_SIZE - 1); // south wall
    addWallSegment(walls, 0, GRID_SIZE - 1, 0, 0);                         // west wall
    addWallSegment(walls, 0, GRID_SIZE - 1, GRID_SIZE - 1, GRID_SIZE - 1); // east wall
    addDoorway(walls, GRID_SIZE - 1, GRID_SIZE - 1, 23, 26);               // front door, opens onto hallway

    // west interior wall -- separates the hallway (j 23-26) from the west
    // rooms (bedroom north, living room south)
    addWallSegment(walls, 1, GRID_SIZE - 2, 22, 22);
    addDoorway(walls, 10, 13, 22, 22);   // bedroom doorway
    addDoorway(walls, 33, 36, 22, 22);   // living room doorway

    // east interior wall -- separates the hallway from the east rooms
    // (bathroom north, kitchen south)
    addWallSegment(walls, 1, GRID_SIZE - 2, 27, 27);
    addDoorway(walls, 5, 8, 27, 27);     // bathroom doorway
    addDoorway(walls, 33, 36, 27, 27);   // kitchen doorway

    // west rooms divider -- bedroom (north) vs living room (south); no
    // direct doorway between them, they're only connected via the hallway
    addWallSegment(walls, 25, 25, 1, 21);

    // east rooms divider -- bathroom (north, smaller) vs kitchen (south)
    addWallSegment(walls, 15, 15, 28, GRID_SIZE - 2);

    return walls;
}


// Returns the concentration value to use for neighbor (ni, nj) when updating
// cell (i, j). Implements a REFLECTING (no-flux) boundary: if the neighbor
// is a wall or off the edge of the grid, gas can't pass through it, so we
// treat it as if it mirrors the current cell's own value. That keeps the
// gradient (and therefore flux) across that face at zero -- physically,
// "nothing crosses a solid boundary".
double neighborValue(const Grid& current, const WallGrid& walls,
                      int i, int j, int ni, int nj) {
    bool outOfBounds = (ni < 0 || ni >= GRID_SIZE || nj < 0 || nj >= GRID_SIZE);
    if (outOfBounds || walls[ni][nj]) {
        return current[i][j];   // reflect: neighbor "mirrors" this cell
    }
    return current[ni][nj];
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

// Builds a folder name like "data/run_2026-08-11_14-32-01" from the current
// wall-clock time.
std::string makeRunFolderName() {
    auto now = std::chrono::system_clock::now();
    std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTime = *std::localtime(&nowTime);

    std::ostringstream oss;
    oss << "data/run_" << std::put_time(&localTime, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

// Writes one grid snapshot to a CSV file: one row per grid row, comma-separated.
// `step` is included in the filename so each snapshot gets its own file --
// simplest possible approach for now, easy to load in Python later.
void writeSnapshot(const Grid& grid, int step, const std::string& runDir) {
    std::string filename = runDir + "/snapshot_" + std::to_string(step) + ".csv";
    std::ofstream out(filename);

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            out << grid[i][j];
            if (j < GRID_SIZE - 1) out << ",";
        }
        out << "\n";
    }
    out.close();
}

// Writes the wall layout to its own CSV file, so we can visualize the house geometry in Python later.
// (Now written inside runDir so it stays grouped with the snapshots from the same run.)
void writeWalls(const WallGrid& walls, const std::string& runDir) {
    std::ofstream out(runDir + "/walls.csv");
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            out << (walls[i][j] ? 1 : 0);
            if (j < GRID_SIZE - 1) out << ",";
        }
        out << "\n";
    }
}

int main() {
    // std::filesystem::create_directories makes the folder (and any parent
    // folders that don't exist yet, e.g. "data/" itself) in one call.
    std::string runDir = makeRunFolderName();
    std::filesystem::create_directories(runDir);
    std::cout << "Writing this run's output to " << runDir << "\n";

    Grid current = makeEmptyGrid();
    Grid next = makeEmptyGrid();
    WallGrid walls = makeHouseLayout();
    writeWalls(walls, runDir);

    for (int step = 0; step < NUM_STEPS; step++) {
        // Inject concentration at the source cell BEFORE diffusing, so the
        // source keeps "topping up" concentration each step rather than
        // just diffusing away a single initial pulse.
        current[SOURCE_X][SOURCE_Y] += SOURCE_RATE;

        diffuseStep(current, walls, next);

        // Swap: `next` becomes the new `current` for the following
        // iteration. std::swap just exchanges what the two variables point
        // to internally.
        std::swap(current, next);

        if (step % LOG_EVERY == 0) {
            writeSnapshot(current, step, runDir);
            std::cout << "Logged step " << step << "\n";
        }
    }

    std::cout << "Done. " << (NUM_STEPS / LOG_EVERY) << " snapshots written to " << runDir << "/\n";
    return 0;
}