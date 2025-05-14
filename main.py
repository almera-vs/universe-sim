"""
Main entry point for the universe simulation.
"""
import random
import logging
import argparse
from simulation.runner import SimulationRunner, test_planet_spacing

def main():
    """Run the simulation."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run universe simulation')
    parser.add_argument('--ticks', type=int, default=100, help='Number of ticks to run')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with detailed output')
    parser.add_argument('--interval', type=int, default=10, help='Output interval (0 for minimal output)')
    parser.add_argument('--visualize', action='store_true', help='Show 2D visualization of the simulation')
    parser.add_argument('--test-spacing', action='store_true', help='Run the planet spacing test')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run planet spacing test if requested
    if args.test_spacing:
        test_planet_spacing()
        return
    
    # Create and run simulation
    runner = SimulationRunner(debug=args.debug, ticks_per_simulation=args.ticks)
    
    # Print information about run mode
    if args.visualize:
        print(f"Running simulation with VISUALIZATION for {args.ticks} ticks")
    else:
        print(f"Running FAST simulation for {args.ticks} ticks")
    
    # Run the simulation with the appropriate visualization setting
    runner.simulate(
        seed=args.seed, 
        ticks=args.ticks, 
        output_interval=args.interval,
        visualize=args.visualize
    )

if __name__ == "__main__":
    main()
