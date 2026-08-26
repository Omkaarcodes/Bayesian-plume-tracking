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
#include <random>


const int GRID_SIZE = 50;
const double D = 1.0;
const double DX = 1.0;
const double DT = 0.2;
const int NUM_STEPS = 600;

const int LOG_RATE = 15;

const int SOURCE_X = 35;
const int SOURCE_Y = 38;
const double SOURCE_RATE = 5.0;

const double SENSOR_NOISE_STD = 0.15; // standard deviation of Gaussian noise added to sensor readings

using Grid = std::vector<std::vector<double>>;


using WallGrid = std::vector<std::vector<bool>>;


struct Sensor {
    int id;
    int x;
    int y;
};


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


void addWallSegment(WallGrid& walls, int iStart, int iEnd, int jStart, int jEnd) {
    for (int i = iStart; i <= iEnd; i++) {
        for (int j = jStart; j <= jEnd; j++) {
            walls[i][j] = true;
        }
    }
}

// Clears a rectangular gap in a wall so two spaces connect.
void addDoorway(WallGrid& walls, int iStart, int iEnd, int jStart, int jEnd) {
    for (int i = iStart; i <= iEnd; i++) {
        for (int j = jStart; j <= jEnd; j++) {
            walls[i][j] = false;
        }
    }
}


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
    addWallSegment(walls, 25, 25, 1, 21);

    // east rooms divider -- bathroom (north, smaller) vs kitchen (south)
    addWallSegment(walls, 15, 15, 28, GRID_SIZE - 2);

    return walls;
}



double neighborValue(const Grid& current, const WallGrid& walls,
                      int i, int j, int ni, int nj) {
    bool outOfBounds = (ni < 0 || ni >= GRID_SIZE || nj < 0 || nj >= GRID_SIZE);
    if (outOfBounds || walls[ni][nj]) {
        return current[i][j];   // reflect: neighbor "mirrors" this cell
    }
    return current[ni][nj];
}


void diffuseStep(const Grid& current, const WallGrid& walls, Grid& next) {
    double alpha = D * DT / (DX * DX); // the D*dt/dx^2 factor from the formula

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            // Wall cells never hold gas -- keep them pinned at 0 and skip
            // the update entirely.
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


std::string makeRunFolderName() {
    auto now = std::chrono::system_clock::now();
    std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTime = *std::localtime(&nowTime);

    std::ostringstream oss;
    oss << "data/run_" << std::put_time(&localTime, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}


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

void writeSensorPositions(const std::vector<Sensor>& sensors, const std::string& runDir) {
    std::ofstream out(runDir + "/sensor_positions.csv");
    out << "sensor_id,x,y\n";
    for (const Sensor& s : sensors) {
        out << s.id << "," << s.x << "," << s.y << "\n";
    }
}
// Samples every sensor's true concentration at its grid cell, adds Gaussian
// noise.

void sampleAndLogSensors(const Grid& current, const std::vector<Sensor>& sensors,
                          int step, std::ofstream& out, std::mt19937& rng) {
    std::normal_distribution<double> noise(0.0, SENSOR_NOISE_STD);

    for (const Sensor& s : sensors) {
        double trueConcentration = current[s.x][s.y];
        double reading = trueConcentration + noise(rng);
        out << step << "," << s.id << "," << trueConcentration << "," << reading << "\n";
    }
}
int main() {

    std::string runDir = makeRunFolderName();
    std::filesystem::create_directories(runDir);
    std::cout << "Writing this run's output to " << runDir << "\n";

    Grid current = makeEmptyGrid();
    Grid next = makeEmptyGrid();
    WallGrid walls = makeHouseLayout();
    writeWalls(walls, runDir);
    std::vector<Sensor> sensors = placeSensors();
    writeSensorPositions(sensors, runDir);

    std::mt19937 rng(std::random_device{}());
    
    std::ofstream sensorLog(runDir + "/sensors.csv");
    sensorLog << "step,sensor_id,true_concentration,reading\n";

    for (int step = 0; step < NUM_STEPS; step++) {
        // Inject concentration at the source cell before diffusing.
        current[SOURCE_X][SOURCE_Y] += SOURCE_RATE;

        diffuseStep(current, walls, next);

        std::swap(current, next);

        sampleAndLogSensors(current, sensors, step, sensorLog, rng);

        if (step % LOG_RATE == 0) {
            writeSnapshot(current, step, runDir);
            std::cout << "Logged step " << step << "\n";
        }
    }

    std::cout << "Done. " << (NUM_STEPS / LOG_RATE) << " snapshots written to " << runDir << "/\n";
    return 0;
}