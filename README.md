# SLAM Problem

## Overview
Implementation of a Simultaneous Localization and Mapping (SLAM) solution using factor graphs. You'll work with sensor data to estimate a robot's trajectory and build a map of its environment simultaneously.

## Problem Description
SLAM is a fundamental challenge in robotics where a robot must determine its own location while constructing a map of an unknown environment. This assignment focuses on the **factor graph representation**, which models SLAM as a probabilistic inference problem where variables (poses and landmarks) are connected through measurement constraints.

## What You'll Do
- Parse and analyze provided sensor data (odometry, landmark observations)
- Build a factor graph that represents the robot's motion model and measurement constraints
- Optimize the factor graph to find the most likely estimates of robot poses and landmark positions
- Evaluate your solution's accuracy against ground truth data

## Expected Outcomes
A working SLAM system that demonstrates pose estimation and map construction with quantified accuracy metrics.
