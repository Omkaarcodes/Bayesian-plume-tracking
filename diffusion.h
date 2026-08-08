//
// Created by Omkaar Sampigeadi on 8/8/2026.
//

#ifndef BAYESIAN_PLUME_TRACKING_DIFFUSION_H
#define BAYESIAN_PLUME_TRACKING_DIFFUSION_H


#include <iostream>
#include <fstream>
#include <vector>

// ---- Simulation parameters ----
// These are the "knobs" of the simulation. Keeping them as named constants
// (instead of magic numbers scattered through the code) makes the physics
// easy to tune later and easy to explain in your README.

const int GRID_SIZE = 50;      // grid is GRID_SIZE x GRID_SIZE cells
const double D = 1.0;          // diffusion coefficient
const double DX = 1.0;         // spacing between grid cells
const double DT = 0.2;         // time step size
const int NUM_STEPS = 600;     // how many timesteps to simulate (longer, so gas has
                                // time to reach the doorway and leak into the 2nd room)
const int LOG_EVERY = 15;      // only write a snapshot every N steps (keeps CSV count sane)

const int SOURCE_X = 25;       // where the source sits: centered in the LEFT room
const int SOURCE_Y = 12;       // (wall/doorway is at column 25, so this is clearly left of it)
const double SOURCE_RATE = 5.0; // how much concentration the source injects each step

